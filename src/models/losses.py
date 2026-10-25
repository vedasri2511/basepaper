"""
Loss Functions for HolisticRCA.
Aligned with paper: "Holistic Root Cause Analysis for Failures in Cloud-Native Systems Through Observability Data"
(Yongqi Han et al., IEEE TSC 2024).

Section IV.E: Multi-Label Fault Classification and Optimization
Eq. (6):
    l_{i, j, c} = - p_c * y_{i, j, c} * log(y_hat_{i, j, c}) - (1 - y_{i, j, c}) * log(1 - y_hat_{i, j, c})
    where p_c = N_{non-faulty, c} / N_{faulty, c}

Eq. (7):
    L_cls = (1 / kappa) * sum_{i=1}^kappa (1 / d_tilde_i) * sum_{j=1}^{d_tilde_i} sum_{c in F_i} l_{i, j, c}
    where:
        kappa: number of entity types (e.g. service, pod, node)
        d_tilde_i: number of entities of entity type i
        F_i: set of fault types associated with entity type i
"""

from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Paper-aligned Weighted Binary Cross-Entropy Loss (Section IV.E, Eq. 6 & 7).

    Operates directly on unnormalized logits for numerical stability.
    """
    def __init__(
        self,
        pos_weights: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None
    ):
        super().__init__()
        self.pos_weights = pos_weights

    def forward(
        self,
        predictions: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: Union[torch.Tensor, Dict[str, torch.Tensor]],
        pos_weights: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None
    ) -> torch.Tensor:
        """
        Computes the paper-defined weighted classification loss across entity types.

        Args:
            predictions: Unnormalized logits tensor or dict of {entity_type: logits_tensor}.
                         Tensor shape: (..., num_entities, num_classes)
            targets: Binary ground-truth target tensor or dict of {entity_type: target_tensor}.
                     Must match the shape of predictions.
            pos_weights: Optional class weights tensor or dict. Overrides self.pos_weights if provided.

        Returns:
            Scalar loss tensor satisfying Eq. (7).
        """
        weights = pos_weights if pos_weights is not None else self.pos_weights

        # Case A: Single tensor prediction and target (kappa = 1)
        if isinstance(predictions, torch.Tensor):
            if not isinstance(targets, torch.Tensor):
                raise TypeError("targets must be a torch.Tensor when predictions is a torch.Tensor.")
            if predictions.shape != targets.shape:
                raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")

            weight_tensor = weights if isinstance(weights, torch.Tensor) else None
            if weight_tensor is not None:
                weight_tensor = weight_tensor.to(device=predictions.device, dtype=predictions.dtype)

            # Element-wise weighted BCE per Eq. (6)
            loss_elements = F.binary_cross_entropy_with_logits(
                predictions,
                targets,
                pos_weight=weight_tensor,
                reduction='none'
            ) # (..., num_entities, num_classes)

            # Eq. (7): sum over classes c in F_i
            loss_per_entity = loss_elements.sum(dim=-1) # (..., num_entities)

            # Eq. (7): average over entities d_tilde_i
            loss_per_entity_type = loss_per_entity.mean(dim=-1) # (...)

            # Average over batch if batched
            return loss_per_entity_type.mean()

        # Case B: Multi-entity type dictionary of predictions and targets (kappa >= 1)
        if not isinstance(predictions, dict) or not isinstance(targets, dict):
            raise TypeError("predictions and targets must both be dicts or both be tensors.")

        entity_losses = []
        for node_type, pred in predictions.items():
            if node_type not in targets:
                raise KeyError(f"Target dictionary missing key '{node_type}' found in predictions.")

            targ = targets[node_type]
            if pred.shape != targ.shape:
                raise ValueError(
                    f"Shape mismatch for node_type '{node_type}': "
                    f"predictions {pred.shape} vs targets {targ.shape}"
                )

            # Determine class weights for this entity type
            w = None
            if isinstance(weights, dict) and node_type in weights:
                w = weights[node_type]
            elif isinstance(weights, torch.Tensor):
                w = weights

            if w is not None:
                w = w.to(device=pred.device, dtype=pred.dtype)

            # Element-wise weighted BCE per Eq. (6)
            loss_elements = F.binary_cross_entropy_with_logits(
                pred,
                targ,
                pos_weight=w,
                reduction='none'
            ) # (..., num_entities, num_classes)

            # Eq. (7): sum over classes c in F_i
            loss_per_entity = loss_elements.sum(dim=-1) # (..., num_entities)

            # Eq. (7): average over entities d_tilde_i
            loss_type_i = loss_per_entity.mean(dim=-1) # (...)

            if loss_type_i.dim() > 0:
                loss_type_i = loss_type_i.mean()

            entity_losses.append(loss_type_i)

        if not entity_losses:
            raise ValueError("No entity losses were computed from predictions and targets.")

        # Eq. (7): average over kappa entity types
        total_loss = torch.stack(entity_losses).mean()
        return total_loss
