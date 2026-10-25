import math
from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding as cited in paper ([59] Vaswani et al. 2017).
    Section IV.B: 'the fixed positional embedding [59] is first added to X to incorporate
    relative positional information, enabling Transformer to learn temporal dependencies.'

    For input matrix X in R^(t x n) where rows represent time buckets and columns
    represent feature channels:
        PE(pos, 2i)   = sin(pos / 10000^(2i / n))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / n))
    This produces a unique geometric progression of sinusoidal frequencies across all
    feature channels rather than broadcasting a single 1D scalar across columns.
    """
    def __init__(self, max_len: int = 5000):
        super().__init__()
        self.max_len = max_len
        self._cache = {}

    def get_pe(self, t: int, d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Retrieves or generates precomputed fixed sinusoidal positional encoding tensor of shape (t, d).
        """
        key = (t, d, str(device), str(dtype))
        if key in self._cache:
            return self._cache[key]

        pe = torch.zeros(t, d, device=device, dtype=dtype)
        position = torch.arange(0, t, device=device, dtype=dtype).unsqueeze(1) # (t, 1)

        num_sin = (d + 1) // 2
        num_cos = d // 2

        div_term = torch.exp(
            torch.arange(0, d, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(1, d))
        )
        pe[:, 0::2] = torch.sin(position * div_term[:num_sin])
        if num_cos > 0:
            pe[:, 1::2] = torch.cos(position * div_term[:num_cos])

        self._cache[key] = pe
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds temporal positional encodings to modality tensor.
        x: Tensor of shape (time_buckets, num_features) or (batch, time_buckets, num_features)
        """
        if x.dim() == 2:
            t, d = x.shape
            pe = self.get_pe(t, d, x.device, x.dtype)
            return x + pe
        elif x.dim() == 3:
            b, t, d = x.shape
            pe = self.get_pe(t, d, x.device, x.dtype)
            return x + pe.unsqueeze(0)
        return x



class ObservabilityRepresentation(nn.Module):
    """
    Section IV.B: Observability Representation Learning
    Eq (1): H = TE_mo(Linear_mo(X'))

    1. Fixed positional embedding added to X along time dimension t.
    2. Transpose position-encoded X to X' in R^(n_features x t).
    3. Project time dimension t to vector space lambda using Linear_mo: R^(n_features x lambda).
    4. Apply stacked Transformer Encoder (TE_mo) across features as graph nodes.
    Output H in R^(n_features x lambda).
    """
    def __init__(self, num_time_buckets: int, embed_dim: int = 128, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.num_time_buckets = num_time_buckets
        self.embed_dim = embed_dim
        self.pos_encoder = PositionalEncoding(max_len=max(5000, num_time_buckets + 10))
        self.linear_map = nn.Linear(num_time_buckets, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (time_buckets, num_features) or (batch, time_buckets, num_features)
        Returns:
            H: Tensor of shape (num_features, embed_dim) or (batch, num_features, embed_dim)
        """
        is_2d = (x.dim() == 2)
        if is_2d:
            x = x.unsqueeze(0) # (1, time_buckets, num_features)

        b, t, n = x.shape
        # Handle time dimension mismatch if needed (truncate or pad)
        if t != self.num_time_buckets:
            if t > self.num_time_buckets:
                x = x[:, :self.num_time_buckets, :]
            else:
                padding = torch.zeros(b, self.num_time_buckets - t, n, device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=1)

        # 1. Add Positional Encoding
        x_pos = self.pos_encoder(x)

        # 2. Transpose to (batch, num_features, time_buckets)
        x_t = x_pos.transpose(1, 2)

        # 3. Map time_buckets to embed_dim (lambda)
        x_mapped = self.linear_map(x_t) # (batch, num_features, embed_dim)

        # 4. Pass through Transformer Encoder across features
        H = self.transformer_encoder(x_mapped) # (batch, num_features, embed_dim)

        return H.squeeze(0) if is_2d else H


class ResourceEntityFeatureIntegration(nn.Module):
    """
    Section IV.C: Resource Entity Feature Integration
    Eq (2): v_e = Flatten(Linear_et(TE_et(HI_e)))

    1. HI_e: Concatenated feature embeddings [HI^T || HI^M || HI^L] of shape (n_e, lambda).
    2. Pass through entity-type-specific Transformer Encoder (TE_et) to fuse cross-modality features.
    3. Flatten (n_e * lambda).
    4. Project via Linear_et to standardized dimension nu.
    Output v_e in R^nu.
    """
    def __init__(self, num_features: int, embed_dim: int = 128, out_dim: int = 256, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear_out = nn.Linear(num_features * embed_dim, out_dim)

    def forward(self, HI_e: torch.Tensor) -> torch.Tensor:
        """
        Args:
            HI_e: Shape (num_entities, n_e, embed_dim) or (batch, num_entities, n_e, embed_dim)
        Returns:
            v_e: Shape (num_entities, out_dim) or (batch, num_entities, out_dim)
        """
        is_3d = (HI_e.dim() == 3)
        if is_3d:
            HI_e = HI_e.unsqueeze(0) # (1, num_entities, n_e, embed_dim)

        b, num_entities, n_e, d = HI_e.shape
        HI_flat_entities = HI_e.view(b * num_entities, n_e, d)

        # Fuse cross-modality embeddings via Transformer Encoder
        fused = self.transformer_encoder(HI_flat_entities) # (b * num_entities, n_e, d)

        # Flatten features and embedding dimensions
        fused_flat = fused.reshape(b * num_entities, n_e * d)

        # Project to standardized dimension nu
        v_e = self.linear_out(fused_flat) # (b * num_entities, out_dim)
        v_e = v_e.view(b, num_entities, self.out_dim)

        return v_e.squeeze(0) if is_3d else v_e


class HolisticRCA(nn.Module):
    """
    Holistic Root Cause Analysis end-to-end architecture (Yongqi Han et al., IEEE TSC 2024).

    Pipeline Stages:
    1. Observability Representation Learning (Section IV.B, Eq. 1):
       H_trace, H_metric, H_log via PositionalEncoding + Linear + TE_mo
    2. Resource Entity Feature Integration (Section IV.C, Eq. 2):
       Gathers rows from H_total using entity index sets into HI_e.
       Fuses cross-modality features via TE_et + Flatten + Linear_et -> V in R^(|E| x nu)
    3. Resource Entity Fusion (Section IV.D, Eq. 3-4):
       Fuses entity embeddings across topology edges in RERG via GATv2Conv -> V' in R^(|E| x nu)
    4. Multi-Label Classification (Section IV.E, Eq. 5):
       Entity-type specific Linear classifiers -> logits in R^(|E| x |F_et|)
    """
    def __init__(
        self,
        node_types: List[str],
        feature_counts: Dict[str, int],
        num_time_buckets: int,
        embed_dim: int = 128,
        entity_out_dim: int = 256,
        num_fault_classes: int = 3,
        fault_classes: Optional[Union[List[str], Dict[str, List[str]]]] = None
    ):
        super().__init__()
        self.node_types = node_types
        self.feature_counts = feature_counts
        self.num_time_buckets = num_time_buckets
        self.embed_dim = embed_dim
        self.entity_out_dim = entity_out_dim

        # Gate C: Configure fault classes per entity type or globally.
        # Paper Section V.B.1: Dataset C defines 3 fault classes for microservices:
        # ['cpu_load', 'network_delay', 'network_loss'].
        if fault_classes is not None:
            if isinstance(fault_classes, list):
                self.fault_classes = {nt: fault_classes for nt in node_types}
                self.num_fault_classes = len(fault_classes)
            else:
                self.fault_classes = fault_classes
                self.num_fault_classes = max(len(fc) for fc in fault_classes.values())
        else:
            base_classes = ['cpu_load', 'network_delay', 'network_loss']
            if num_fault_classes <= len(base_classes):
                default_classes = base_classes[:num_fault_classes]
            else:
                default_classes = base_classes + [f'fault_class_{i}' for i in range(len(base_classes), num_fault_classes)]
            self.fault_classes = {nt: default_classes for nt in node_types}
            self.num_fault_classes = num_fault_classes

        # 1. Modality-specific representation learners (TE_mo)
        self.trace_rep = ObservabilityRepresentation(num_time_buckets, embed_dim)
        self.metric_rep = ObservabilityRepresentation(num_time_buckets, embed_dim)
        self.log_rep = ObservabilityRepresentation(num_time_buckets, embed_dim)

        # 2. Entity type-specific integrators (TE_et)
        self.entity_integrators = nn.ModuleDict({
            node_type: ResourceEntityFeatureIntegration(
                num_features=feature_counts[node_type],
                embed_dim=embed_dim,
                out_dim=entity_out_dim
            ) for node_type in node_types
        })

        # 3. Resource Entity Fusion using Graph Attention Network (GATv2)
        # Gate B: Equations (3)-(4) in Section IV.D define standard GATv2 aggregation across
        # all edges in E using a shared weight matrix W and attention vector a.
        # RERGBuilder constructs bidirectional call/deployment edges in edge_index.
        # edge_types serves as descriptive metadata for provenance and explainability.
        # add_self_loops=False honors Step 1.3 contract (RERGBuilder already supplies self-loops).
        self.gat = GATv2Conv(
            in_channels=entity_out_dim,
            out_channels=entity_out_dim,
            heads=4,
            concat=False,
            add_self_loops=False
        )

        # 4. Multi-head Fault Classifiers per entity type (Section IV.E, Eq. 5)
        # Produces unnormalized logits; sigmoid is applied at evaluation or inside BCEWithLogitsLoss.
        self.classifiers = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.ELU(),
                nn.Linear(entity_out_dim, len(self.fault_classes[node_type]))
            ) for node_type in node_types
        })



    def extract_modality_representations(
        self,
        traces: Optional[torch.Tensor],
        metrics: Optional[torch.Tensor],
        logs: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes feature embeddings for all active modalities and concatenates them into H_total.
        Output shape: (total_features, embed_dim) or (batch, total_features, embed_dim)
        """
        reps = []
        if traces is not None and traces.numel() > 0:
            reps.append(self.trace_rep(traces))
        if metrics is not None and metrics.numel() > 0:
            reps.append(self.metric_rep(metrics))
        if logs is not None and logs.numel() > 0:
            reps.append(self.log_rep(logs))

        if not reps:
            raise ValueError("At least one observability modality must be provided.")

        # Concatenate along feature dimension (-2)
        H_total = torch.cat(reps, dim=-2)
        return H_total

    def gather_entity_features(
        self,
        H_total: torch.Tensor,
        entity_feature_indices: Union[torch.Tensor, Dict[str, torch.Tensor], Dict[str, List[int]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Gathers entity feature blocks HI_e from H_total using entity mapping index sets.
        Returns:
            Dict mapping node_type -> Tensor of shape (num_entities_of_type, n_e, embed_dim)
        """
        HI_dict = {}

        # Case A: Tensor of shape (num_nodes, n_e)
        if isinstance(entity_feature_indices, torch.Tensor):
            node_type = self.node_types[0]
            if H_total.dim() == 2:
                HI_dict[node_type] = H_total[entity_feature_indices]
            else:
                HI_dict[node_type] = H_total[:, entity_feature_indices, :]
            return HI_dict

        # Case B: Dictionary of node_type -> index tensor or list of indices
        for node_type in self.node_types:
            if node_type not in entity_feature_indices:
                continue
            idx = entity_feature_indices[node_type]
            if not isinstance(idx, torch.Tensor):
                idx = torch.tensor(idx, dtype=torch.long, device=H_total.device)
            else:
                idx = idx.to(device=H_total.device)

            if H_total.dim() == 2:
                HI_dict[node_type] = H_total[idx]
            else:
                HI_dict[node_type] = H_total[:, idx, :]

        return HI_dict

    def get_fused_entity_embeddings(
        self,
        traces: Optional[torch.Tensor],
        metrics: Optional[torch.Tensor],
        logs: Optional[torch.Tensor],
        entity_feature_indices: Union[torch.Tensor, Dict[str, Any]],
        edge_index: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Executes representation learning, entity integration, and GAT fusion.
        Supports optional edge_mask for Section IV.F paper-faithful mask learning.
        Returns:
            v_prime: Topology-fused entity embeddings of shape (total_nodes, entity_out_dim)
        """
        # 1. Representation Learning (Eq. 1)
        H_total = self.extract_modality_representations(traces, metrics, logs)

        # 2. Entity Integration (Eq. 2)
        HI_dict = self.gather_entity_features(H_total, entity_feature_indices)

        entity_embeddings = []
        for node_type in self.node_types:
            if node_type in HI_dict:
                HI_et = HI_dict[node_type]
                v_et = self.entity_integrators[node_type](HI_et)
                entity_embeddings.append(v_et)

        if not entity_embeddings:
            raise ValueError("No entity embeddings were constructed.")

        # Standardized node matrix V in R^(|E| x nu)
        v = torch.cat(entity_embeddings, dim=-2)

        # 3. Resource Entity Fusion (Eq. 3-4)
        edge_index = edge_index.to(device=v.device)

        # Connect edge_mask to GATv2Conv message passing if provided
        if edge_mask is not None:
            self.gat.explain = True
            self.gat._edge_mask = edge_mask
            self.gat._loop_mask = (edge_index[0] != edge_index[1])
            self.gat._apply_sigmoid = True

        try:
            if v.dim() == 3:
                # Batched input: apply GAT message passing per batch sample
                b = v.shape[0]
                v_prime_list = [self.gat(v[i], edge_index) for i in range(b)]
                v_prime = torch.stack(v_prime_list, dim=0)
            else:
                v_prime = self.gat(v, edge_index)
        finally:
            if edge_mask is not None:
                self.gat.explain = False
                self.gat._edge_mask = None
                self.gat._loop_mask = None
                self.gat._apply_sigmoid = True

        return v_prime

    def forward(
        self,
        traces: Optional[torch.Tensor],
        metrics: Optional[torch.Tensor],
        logs: Optional[torch.Tensor],
        entity_feature_indices: Union[torch.Tensor, Dict[str, Any]],
        edge_index: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Complete HolisticRCA forward pass returning classification logits per entity type.
        """
        v_prime = self.get_fused_entity_embeddings(
            traces=traces,
            metrics=metrics,
            logs=logs,
            entity_feature_indices=entity_feature_indices,
            edge_index=edge_index,
            edge_mask=edge_mask
        )

        # 4. Multi-classifier predictions (Eq. 5)
        predictions = {}
        start_idx = 0
        is_batched = (v_prime.dim() == 3)

        # Slice entities per node type and compute logits
        for node_type in self.node_types:
            if isinstance(entity_feature_indices, torch.Tensor):
                count = entity_feature_indices.shape[0] if not is_batched else entity_feature_indices.shape[1]
            elif isinstance(entity_feature_indices, dict) and node_type in entity_feature_indices:
                idx_obj = entity_feature_indices[node_type]
                count = len(idx_obj) if isinstance(idx_obj, list) else idx_obj.shape[0]
            else:
                count = 0

            if count > 0:
                end_idx = start_idx + count
                v_et_prime = v_prime[..., start_idx:end_idx, :]
                predictions[node_type] = self.classifiers[node_type](v_et_prime)
                start_idx = end_idx

        return predictions

    def predict_probabilities(
        self,
        predictions_or_logits: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Applies sigmoid activation to raw logits according to paper Eq. (5):
            y_hat_{i, c} = sigmoid(W_et * v'_{e_i} + b_et)_c
        """
        if isinstance(predictions_or_logits, torch.Tensor):
            return torch.sigmoid(predictions_or_logits)
        return {k: torch.sigmoid(v) for k, v in predictions_or_logits.items()}
