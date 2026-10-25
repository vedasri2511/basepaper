import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EntityRootCauseScore:
    """
    Score attribution for candidate root cause entity (Section IV.F, V.B.3).
    Combines topological relation mask significance and entity feature mask significance.
    """
    entity_name: str
    entity_idx: int
    score: float
    relation_score: float
    feature_score: float
    predicted_probs: Dict[str, float]
    predicted_fault: Optional[str]


@dataclass
class FeatureAttributionScore:
    """
    Observability feature attribution score (Section IV.F, V.G).
    """
    feature_name: str
    feature_idx: int
    modality: str
    entity_name: Optional[str]
    score: float


@dataclass
class EdgeExplanationScore:
    """
    Learned relation mask attribution for an edge in RERG (Section IV.F).
    """
    source_entity: str
    target_entity: str
    source_idx: int
    target_idx: int
    edge_weight: float


@dataclass
class ExplanationOutput:
    """
    Comprehensive, machine-readable explanation result for an observation window.
    Designed for research auditability, automated evaluation, and downstream API consumption.
    """
    timestamp: int
    target_entity: Optional[str]
    target_entity_idx: Optional[int]
    is_fault_predicted: bool
    predicted_fault_types: Dict[str, str]
    max_fault_probability: float
    root_cause_ranking: List[Dict[str, Any]]
    top_feature_attributions: List[Dict[str, Any]]
    edge_explanations: List[Dict[str, Any]]
    optimization_stats: Dict[str, Any]
    fidelity_metrics: Dict[str, float]
    limitations_and_warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MaskLearner(nn.Module):
    """
    Section IV.F: Mask Learning (Equations 8).

    Learns continuous relation mask EM in R^(n_g x n_g) and observability feature mask om in R^(n_f)
    to identify the compact subgraph and subset of features that best explain the model's prediction.

    The optimization objective maximizes mutual information between original predictions and
    masked predictions while regularizing mask sparsity (size) and discrete decisiveness (entropy).

    Model parameters are strictly frozen throughout mask learning.
    """
    def __init__(
        self,
        trained_model: nn.Module,
        num_nodes: int,
        feature_counts: Dict[str, int],
        init_strategy: str = "normal",
        seed: Optional[int] = None
    ):
        """
        Args:
            trained_model: Pre-trained HolisticRCA model (will be frozen)
            num_nodes: Total entities n_g in RERG
            feature_counts: Dict mapping modality name ('trace', 'metric', 'log') to feature count
            init_strategy: 'normal' (small Gaussian perturbation around 0) or 'uniform'
            seed: Optional random seed for reproducible mask initialization
        """
        super().__init__()
        self.model = trained_model
        self.num_nodes = num_nodes
        self.feature_counts = feature_counts

        # Strictly freeze all parameters of the saved classifier
        for param in self.model.parameters():
            param.requires_grad = False

        self.n_trace = feature_counts.get('trace', 0)
        self.n_metric = feature_counts.get('metric', 0)
        self.n_log = feature_counts.get('log', 0)
        self.total_features = self.n_trace + self.n_metric + self.n_log

        if seed is not None:
            torch.manual_seed(seed)

        # Learnable mask for resource entity relations in RERG (Eq 8: EM in R^(n_g x n_g))
        if init_strategy == "normal":
            self.EM = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.05)
            self.om = nn.Parameter(torch.randn(self.total_features) * 0.05)
        elif init_strategy == "zeros":
            self.EM = nn.Parameter(torch.zeros(num_nodes, num_nodes))
            self.om = nn.Parameter(torch.zeros(self.total_features))
        elif init_strategy == "ones":
            self.EM = nn.Parameter(torch.ones(num_nodes, num_nodes) * 0.5)
            self.om = nn.Parameter(torch.ones(self.total_features) * 0.5)
        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")

    def get_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the bounded masks in [0, 1] via sigmoid:
        sigma(EM) for entity relations, sigma(om) for observability features.
        """
        relation_mask = torch.sigmoid(self.EM)
        feature_mask = torch.sigmoid(self.om)
        return relation_mask, feature_mask

    def forward(
        self,
        traces: Optional[torch.Tensor],
        metrics: Optional[torch.Tensor],
        logs: Optional[torch.Tensor],
        entity_mappings: Union[torch.Tensor, Dict[str, Any]],
        edge_index: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Executes masked forward pass through HolisticRCA:
        1. Modulates observability features X \odot (om . 1^T) per modality partition
        2. Modulates RERG message passing G \odot \sigma(EM) via edge_mask

        Returns:
            masked_predictions: Output logits dict from HolisticRCA
            edge_mask: Sigmoid mask values evaluated at edge_index (num_edges,)
            feature_mask: Sigmoid mask values for all observability features (total_features,)
        """
        relation_mask, feature_mask = self.get_masks()

        # 1. Observability Feature Masking (X \odot om)
        # Slices om according to canonical modality partitions
        fm_trace = feature_mask[:self.n_trace]
        fm_metric = feature_mask[self.n_trace:self.n_trace + self.n_metric]
        fm_log = feature_mask[self.n_trace + self.n_metric:self.total_features]

        masked_traces = self._apply_feature_mask(traces, fm_trace) if traces is not None else None
        masked_metrics = self._apply_feature_mask(metrics, fm_metric) if metrics is not None else None
        masked_logs = self._apply_feature_mask(logs, fm_log) if logs is not None else None

        # 2. Topology Masking (G \odot \sigma(EM))
        # Evaluates relation mask values along existing graph edges
        row, col = edge_index[0], edge_index[1]
        edge_mask = relation_mask[row, col]

        # 3. Pass through frozen HolisticRCA model with edge_mask modulating GATv2Conv
        masked_predictions = self.model(
            traces=masked_traces,
            metrics=masked_metrics,
            logs=logs if masked_logs is None else masked_logs,
            entity_feature_indices=entity_mappings,
            edge_index=edge_index,
            edge_mask=edge_mask
        )

        return masked_predictions, edge_mask, feature_mask

    @staticmethod
    def _apply_feature_mask(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Applies feature mask broadcasting across batch and temporal window dimensions.
        """
        if tensor is None or tensor.numel() == 0 or mask.numel() == 0:
            return tensor

        if tensor.dim() == 2:
            # Shape: (window_size, num_features)
            return tensor * mask.unsqueeze(0)
        elif tensor.dim() == 3:
            # Shape: (batch, window_size, num_features)
            return tensor * mask.view(1, 1, -1)
        elif tensor.dim() == 1:
            return tensor * mask
        else:
            raise ValueError(f"Unexpected tensor dim: {tensor.dim()}")

    def loss_function(
        self,
        masked_preds: Dict[str, torch.Tensor],
        original_preds: Dict[str, torch.Tensor],
        edge_mask: torch.Tensor,
        feature_mask: torch.Tensor,
        size_reg: float = 0.005,
        entropy_reg: float = 0.001,
        target_entity_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates optimization objective according to Section IV.F (Eq. 8):
        max MI(y_hat, y_hat_masked) - lambda_size * L_size - lambda_ent * L_ent

        Per GNNExplainer (Ying et al., NeurIPS 2019, ref [66]), maximizing mutual information
        is formulated as minimizing conditional entropy H(Y | G_sub, X_sub) between unmasked
        probabilities sigma(logits_orig) and masked logits via binary cross-entropy:
            L_MI = BCEWithLogits(logits_masked, sigma(logits_orig))

        Regularization terms:
            L_size = mean(edge_mask) + mean(feature_mask)
            L_ent  = mean(H(edge_mask)) + mean(H(feature_mask))
        where H(p) = -p log(p + eps) - (1-p) log(1-p + eps).
        """
        eps = 1e-12
        mi_loss = torch.tensor(0.0, device=edge_mask.device)

        for node_type, m_logits in masked_preds.items():
            if node_type not in original_preds:
                continue
            orig_logits = original_preds[node_type]
            orig_target_prob = torch.sigmoid(orig_logits).detach()

            if target_entity_idx is not None:
                # Target single suspect/faulty entity
                if m_logits.dim() == 3:
                    m_sub = m_logits[:, target_entity_idx, :]
                    o_sub = orig_target_prob[:, target_entity_idx, :]
                else:
                    m_sub = m_logits[target_entity_idx:target_entity_idx + 1, :]
                    o_sub = orig_target_prob[target_entity_idx:target_entity_idx + 1, :]
                mi_loss = mi_loss + F.binary_cross_entropy_with_logits(m_sub, o_sub)
            else:
                # Holistic explanation over entire RERG
                mi_loss = mi_loss + F.binary_cross_entropy_with_logits(m_logits, orig_target_prob)

        # Sparsity / size regularization (prefer minimal, compact subgraphs and feature subsets)
        size_loss = edge_mask.mean() + feature_mask.mean()

        # Entropy regularization (drives soft mask probabilities toward discrete 0 or 1)
        e_ent = -(edge_mask * torch.log(edge_mask + eps) + (1.0 - edge_mask) * torch.log(1.0 - edge_mask + eps))
        f_ent = -(feature_mask * torch.log(feature_mask + eps) + (1.0 - feature_mask) * torch.log(1.0 - feature_mask + eps))
        entropy_loss = e_ent.mean() + f_ent.mean()

        total_loss = mi_loss + size_reg * size_loss + entropy_reg * entropy_loss

        loss_components = {
            'total_loss': float(total_loss.item()),
            'mi_loss': float(mi_loss.item()),
            'size_loss': float(size_loss.item()),
            'entropy_loss': float(entropy_loss.item())
        }
        return total_loss, loss_components


class HolisticRCAExplainer:
    """
    Production-grade, reusable explanation interface for HolisticRCA.

    Generates paper-faithful root-cause explanations by executing mask learning (Section IV.F),
    ranking candidate root-cause entities (Section V.B.3 / V.G), and attributing anomalous
    observability features.
    """
    def __init__(
        self,
        model: nn.Module,
        entities: List[str],
        fault_classes: List[str],
        feature_counts: Dict[str, int],
        feature_names: Optional[Dict[str, List[str]]] = None,
        entity_feature_indices: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: Trained HolisticRCA model
            entities: Canonical ordered list of entity names
            fault_classes: Canonical list of fault types (e.g. ['cpu_load', 'network_delay', 'network_loss'])
            feature_counts: Dict with keys 'trace', 'metric', 'log' indicating feature counts
            feature_names: Optional Dict mapping modality -> list of feature column names
            entity_feature_indices: Optional gathering indices tensor or dict for entity features
            device: torch device (CPU or GPU)
        """
        self.model = model
        self.entities = entities
        self.fault_classes = fault_classes
        self.feature_counts = feature_counts
        self.feature_names = feature_names or {}
        self.entity_feature_indices = entity_feature_indices
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_nodes = len(entities)
        self.model.to(self.device)
        self.model.eval()

    def explain_window(
        self,
        traces: torch.Tensor,
        metrics: torch.Tensor,
        logs: torch.Tensor,
        edge_index: torch.Tensor,
        entity_indices: Optional[torch.Tensor] = None,
        timestamp: int = 0,
        target_entity: Optional[str] = None,
        epochs: int = 100,
        lr: float = 0.05,
        size_reg: float = 0.005,
        entropy_reg: float = 0.001,
        detection_threshold: float = 0.5,
        seed: Optional[int] = 42
    ) -> ExplanationOutput:
        """
        Performs mask optimization and generates structured explanation output.

        Args:
            traces: Tensor (window_size, n_trace) or (1, window_size, n_trace)
            metrics: Tensor (window_size, n_metric) or (1, window_size, n_metric)
            logs: Tensor (window_size, n_log) or (1, window_size, n_log)
            edge_index: Tensor (2, num_edges)
            entity_indices: Tensor (num_entities, n_e)
            timestamp: Observation window timestamp
            target_entity: Optional name of specific entity to explain. If None, targets
                           the entity with highest fault probability, or whole graph if normal.
            epochs: Mask optimization iterations
            lr: Adam optimizer learning rate
            size_reg: Sparsity regularization weight
            entropy_reg: Discrete entropy regularization weight
            detection_threshold: Probability threshold above which an entity is classified faulty
            seed: Reproducibility seed

        Returns:
            ExplanationOutput with full ranking, feature attribution, and diagnostic metadata.
        """
        # Ensure proper device placement
        traces = traces.to(self.device)
        metrics = metrics.to(self.device)
        logs = logs.to(self.device)
        edge_index = edge_index.to(self.device)

        indices = entity_indices if entity_indices is not None else self.entity_feature_indices
        if indices is None:
            raise ValueError("entity_indices must be provided either at initialization or explain_window.")
        if isinstance(indices, torch.Tensor):
            indices = indices.to(self.device)

        # 1. Unmasked Forward Pass: establish baseline prediction
        with torch.no_grad():
            orig_preds = self.model(traces, metrics, logs, indices, edge_index)
            # Default node_type is 'service'
            node_type = list(orig_preds.keys())[0]
            orig_logits = orig_preds[node_type] # (num_entities, num_classes) or (1, num_entities, num_classes)
            if orig_logits.dim() == 3:
                orig_logits_s = orig_logits.squeeze(0)
            else:
                orig_logits_s = orig_logits
            orig_probs = torch.sigmoid(orig_logits_s) # (num_entities, num_classes)

        # Determine predicted fault status per entity
        max_prob_per_entity, max_class_idx_per_entity = torch.max(orig_probs, dim=-1)
        global_max_prob, global_faulty_entity_idx = torch.max(max_prob_per_entity, dim=0)

        is_fault_predicted = bool((global_max_prob >= detection_threshold).item())
        predicted_fault_types = {}
        for idx, ent_name in enumerate(self.entities):
            prob = float(max_prob_per_entity[idx].item())
            if prob >= detection_threshold:
                c_idx = int(max_class_idx_per_entity[idx].item())
                predicted_fault_types[ent_name] = self.fault_classes[c_idx]
            else:
                predicted_fault_types[ent_name] = "normal"

        # Resolve target entity index
        target_idx: Optional[int] = None
        if target_entity is not None:
            if target_entity in self.entities:
                target_idx = self.entities.index(target_entity)
            else:
                raise ValueError(f"target_entity '{target_entity}' not in entities: {self.entities}")
        elif is_fault_predicted:
            target_idx = int(global_faulty_entity_idx.item())
            target_entity = self.entities[target_idx]

        # 2. Instantiate and optimize MaskLearner
        learner = MaskLearner(
            trained_model=self.model,
            num_nodes=self.num_nodes,
            feature_counts=self.feature_counts,
            init_strategy="normal",
            seed=seed
        ).to(self.device)

        optimizer = torch.optim.Adam([learner.EM, learner.om], lr=lr)

        initial_loss = 0.0
        final_loss = 0.0
        loss_history = []

        learner.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            m_preds, edge_mask, feat_mask = learner(traces, metrics, logs, indices, edge_index)
            loss, loss_dict = learner.loss_function(
                masked_preds=m_preds,
                original_preds=orig_preds,
                edge_mask=edge_mask,
                feature_mask=feat_mask,
                size_reg=size_reg,
                entropy_reg=entropy_reg,
                target_entity_idx=target_idx
            )

            # Check for finite loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Mask learning encountered non-finite loss at epoch {epoch}: {loss.item()}")

            loss.backward()
            optimizer.step()

            if epoch == 0:
                initial_loss = loss_dict['total_loss']
            final_loss = loss_dict['total_loss']
            loss_history.append(loss_dict['total_loss'])

        # 3. Post-optimization evaluation with frozen learned masks
        learner.eval()
        with torch.no_grad():
            final_m_preds, learned_edge_mask, learned_feat_mask = learner(
                traces, metrics, logs, indices, edge_index
            )
            rel_mask, _ = learner.get_masks()

            # Fidelity: L1 difference between masked logits and unmasked logits for target
            orig_t = orig_logits_s[target_idx] if target_idx is not None else orig_logits_s
            final_logits = final_m_preds[node_type]
            if final_logits.dim() == 3:
                final_logits = final_logits.squeeze(0)
            final_t = final_logits[target_idx] if target_idx is not None else final_logits

            fidelity_diff = float(torch.abs(orig_t - final_t).mean().item())
            fidelity_prob_diff = float(torch.abs(torch.sigmoid(orig_t) - torch.sigmoid(final_t)).mean().item())

        # 4. Dimension 1: Root-Cause Entity Ranking (Section IV.F, V.B.3, V.G)
        # Compute relation score (sum of learned edge mask weights incident to each node)
        # and feature score (sum of feature mask values assigned to that entity)
        row, col = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
        edge_weights_np = learned_edge_mask.detach().cpu().numpy()
        feat_mask_np = learned_feat_mask.detach().cpu().numpy()

        incident_edge_scores = [0.0] * self.num_nodes
        for src, dst, w in zip(row, col, edge_weights_np):
            incident_edge_scores[src] += float(w)
            incident_edge_scores[dst] += float(w)

        # Compute entity feature scores
        entity_feature_scores = [0.0] * self.num_nodes
        if isinstance(indices, torch.Tensor):
            indices_np = indices.cpu().numpy()
            for ent_i in range(self.num_nodes):
                ent_feats = indices_np[ent_i]
                entity_feature_scores[ent_i] = float(np.sum(feat_mask_np[ent_feats]))
        else:
            # Fallback to uniform partition if indices dict not standard
            feats_per_ent = max(1, len(feat_mask_np) // self.num_nodes)
            for ent_i in range(self.num_nodes):
                entity_feature_scores[ent_i] = float(np.sum(feat_mask_np[ent_i*feats_per_ent:(ent_i+1)*feats_per_ent]))

        # Combined root-cause score: normalized relation score + normalized feature score
        max_rel = max(incident_edge_scores) if max(incident_edge_scores) > 0 else 1.0
        max_feat = max(entity_feature_scores) if max(entity_feature_scores) > 0 else 1.0

        entity_scores = []
        for ent_i, ent_name in enumerate(self.entities):
            norm_rel = incident_edge_scores[ent_i] / max_rel
            norm_feat = entity_feature_scores[ent_i] / max_feat
            combined_score = 0.5 * norm_rel + 0.5 * norm_feat

            p_dict = {
                fc: float(orig_probs[ent_i, c_i].item())
                for c_i, fc in enumerate(self.fault_classes)
            }
            pred_f = predicted_fault_types[ent_name]

            entity_scores.append(EntityRootCauseScore(
                entity_name=ent_name,
                entity_idx=ent_i,
                score=float(combined_score),
                relation_score=float(incident_edge_scores[ent_i]),
                feature_score=float(entity_feature_scores[ent_i]),
                predicted_probs=p_dict,
                predicted_fault=pred_f if pred_f != "normal" else None
            ))

        # Sort entities descending by combined score
        entity_scores.sort(key=lambda x: x.score, reverse=True)
        root_cause_ranking = [asdict(score) for score in entity_scores]

        # 5. Dimension 2: Observability Feature Attribution (Section IV.F, V.G)
        feature_attributions = []
        all_feature_names = self._build_flat_feature_name_list()

        for f_idx, w in enumerate(feat_mask_np):
            fname, modality, ent = all_feature_names[f_idx]
            feature_attributions.append(FeatureAttributionScore(
                feature_name=fname,
                feature_idx=f_idx,
                modality=modality,
                entity_name=ent,
                score=float(w)
            ))

        # Top 15 features sorted by importance
        feature_attributions.sort(key=lambda x: x.score, reverse=True)
        top_feature_attributions = [asdict(fa) for fa in feature_attributions[:15]]

        # 6. Edge Explanations in RERG
        edge_explanations = []
        for src, dst, w in zip(row, col, edge_weights_np):
            edge_explanations.append(EdgeExplanationScore(
                source_entity=self.entities[src],
                target_entity=self.entities[dst],
                source_idx=int(src),
                target_idx=int(dst),
                edge_weight=float(w)
            ))
        edge_explanations.sort(key=lambda x: x.edge_weight, reverse=True)
        edge_explanations_dict = [asdict(ee) for ee in edge_explanations]

        # 7. Diagnostic and Limitation Notices
        warnings = []
        if not is_fault_predicted:
            warnings.append(
                "Prediction is normal across all entities. Mask learning reflects "
                "subgraph attribution for the background normal state, not a positive failure."
            )
        warnings.append(
            "EXPERIMENTAL RESULT: The current checkpoint was trained on Dataset C slice with a single "
            "continuous cpu_load episode on social_service_3. Explanations reflect this local distribution "
            "and should not be claimed as generalized cloud RCA accuracy."
        )

        return ExplanationOutput(
            timestamp=timestamp,
            target_entity=target_entity,
            target_entity_idx=target_idx,
            is_fault_predicted=is_fault_predicted,
            predicted_fault_types=predicted_fault_types,
            max_fault_probability=float(global_max_prob.item()),
            root_cause_ranking=root_cause_ranking,
            top_feature_attributions=top_feature_attributions,
            edge_explanations=edge_explanations_dict,
            optimization_stats={
                'epochs': epochs,
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'learning_rate': lr,
                'loss_reduction': float(initial_loss - final_loss)
            },
            fidelity_metrics={
                'logit_l1_diff': fidelity_diff,
                'prob_l1_diff': fidelity_prob_diff
            },
            limitations_and_warnings=warnings
        )

    def _build_flat_feature_name_list(self) -> List[Tuple[str, str, Optional[str]]]:
        """
        Builds a canonical (name, modality, entity) tuple for every feature index 0..265.
        """
        names = []
        trace_names = self.feature_names.get('trace', [f"trace_feat_{i}" for i in range(self.feature_counts.get('trace', 0))])
        metric_names = self.feature_names.get('metric', [f"metric_feat_{i}" for i in range(self.feature_counts.get('metric', 0))])
        log_names = self.feature_names.get('log', [f"log_feat_{i}" for i in range(self.feature_counts.get('log', 0))])

        for name in trace_names:
            ent = self._infer_entity_from_feature_name(name)
            names.append((name, 'trace', ent))
        for name in metric_names:
            ent = self._infer_entity_from_feature_name(name)
            names.append((name, 'metric', ent))
        for name in log_names:
            ent = self._infer_entity_from_feature_name(name)
            names.append((name, 'log', ent))

        return names

    def _infer_entity_from_feature_name(self, feature_name: str) -> Optional[str]:
        for ent in self.entities:
            if ent in feature_name:
                return ent
        return None
