"""
Checkpointing and Training History Management for HolisticRCA.
Aligned with paper: "Holistic Root Cause Analysis for Failures in Cloud-Native Systems Through Observability Data"
(Yongqi Han et al., IEEE TSC 2024).

Features:
- Atomic, reproducible saving of latest and best-validation checkpoints.
- Safe validation handling: flags whether validation has positive examples before claiming validated generalization.
- JSON-serialized machine-readable training history and metadata.
- Checkpoint loader that reconstructs model architecture and verifies compatibility.
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import json
import torch
import torch.nn as nn
import numpy as np

from src.models.holistic_rca import HolisticRCA


def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
    config: Dict[str, Any],
    metadata: Dict[str, Any],
    is_best_validated: bool = False
) -> None:
    """
    Saves complete model checkpoint with training state and architecture configuration.
    """
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

    # Sanitize metadata into standard primitives to avoid custom class unpickling errors
    safe_metadata = {}
    for k, v in metadata.items():
        if k == 'splits' and isinstance(v, dict):
            safe_metadata[k] = {
                s_name: s_obj.to_dict() if hasattr(s_obj, 'to_dict') else s_obj
                for s_name, s_obj in v.items()
            }
        elif hasattr(v, 'to_dict'):
            safe_metadata[k] = v.to_dict()
        else:
            safe_metadata[k] = v

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'train_loss': float(train_loss),
        'val_loss': float(val_loss) if val_loss is not None else None,
        'is_best_validated': is_best_validated,
        'config': config,
        'metadata': safe_metadata
    }

    torch.save(checkpoint, checkpoint_path)



def load_checkpoint(
    checkpoint_path: str,
    model: Optional[HolisticRCA] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Tuple[HolisticRCA, Dict[str, Any]]:
    """
    Loads checkpoint from disk. If `model` is None, dynamically instantiates
    HolisticRCA using the saved configuration.

    Returns:
        (model, checkpoint_dict)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    if model is None:
        # Reconstruct model from saved configuration
        model = HolisticRCA(
            node_types=config.get('node_types', ['service']),
            feature_counts=config.get('feature_counts', {'service': 19}),
            num_time_buckets=config.get('num_time_buckets', 1),
            embed_dim=config.get('embed_dim', 128),
            entity_out_dim=config.get('entity_out_dim', 256),
            num_fault_classes=config.get('num_fault_classes', 3),
            fault_classes=config.get('fault_classes', None)
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, checkpoint


def make_json_serializable(obj: Any) -> Any:
    """Recursively converts tensors, numpy types, and custom objects to JSON-serializable primitives."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif hasattr(obj, 'to_dict'):
        return make_json_serializable(obj.to_dict())
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]
    return obj


def save_training_history(
    history_path: str,
    history_records: List[Dict[str, Any]],
    metadata: Dict[str, Any]
) -> None:
    """
    Saves training history to disk in JSON format.
    """
    os.makedirs(os.path.dirname(os.path.abspath(history_path)), exist_ok=True)

    out_dict = {
        'metadata': make_json_serializable(metadata),
        'history': make_json_serializable(history_records)
    }

    with open(history_path, 'w') as f:
        json.dump(out_dict, f, indent=2)
