"""
Training Pipeline for HolisticRCA.
Aligned with paper: "Holistic Root Cause Analysis for Failures in Cloud-Native Systems Through Observability Data"
(Yongqi Han et al., IEEE TSC 2024).

Section IV.E: Multi-Label Fault Classification and Optimization
Eq. (6):
    l_{i, j, c} = - p_c * y_{i, j, c} * log(y_hat_{i, j, c}) - (1 - y_{i, j, c}) * log(1 - y_hat_{i, j, c})
    where p_c = N_{non-faulty, c} / N_{faulty, c}

Eq. (7):
    L_cls = (1 / kappa) * sum_{i=1}^kappa (1 / d_tilde_i) * sum_{j=1}^{d_tilde_i} sum_{c in F_i} l_{i, j, c}

Key features:
- Real dataset integration with Dataset C.
- Strict data leakage prevention (training-only vocabulary and class weights).
- Robust gradient checking and non-finite value protection.
- Reproducible training with fixed seeds.
- Checkpoint and training history persistence.
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add repository root to path if executed directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.dataset import load_dataset_c_splits, collate_holistic_rca
from src.data.label_encoder import DEFAULT_SERVICE_FAULT_CLASSES
from src.models.holistic_rca import HolisticRCA
from src.models.losses import WeightedBCELoss
from src.models.checkpoint import save_checkpoint, save_training_history
from src.evaluate import evaluate_fault_classification


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_holistic_rca(
    data_dir: str = "data_samples/dataset_c",
    checkpoint_dir: str = "checkpoints",
    epochs: int = 10,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    batch_size: int = 4,
    window_size: int = 1,
    embed_dim: int = 128,
    entity_out_dim: int = 256,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Executes end-to-end training and evaluation of HolisticRCA.
    """
    set_seed(seed)
    print("=" * 70)
    print("HOLISTIC RCA TRAINING PIPELINE (IEEE TSC 2024)")
    print("=" * 70)
    print(f"Data Directory:   {data_dir}")
    print(f"Checkpoint Dir:   {checkpoint_dir}")
    print(f"Hyperparameters:  epochs={epochs}, lr={lr}, batch_size={batch_size}, window_size={window_size}")
    print(f"Architecture:     embed_dim={embed_dim}, entity_out_dim={entity_out_dim}, device={device}")

    # 1. Load Real Dataset with Chronological Split & Leakage Prevention
    train_dataset, val_dataset, test_dataset, metadata = load_dataset_c_splits(
        data_dir=data_dir,
        time_granularity=60,
        window_size=window_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        fault_classes=DEFAULT_SERVICE_FAULT_CLASSES
    )

    # Diagnostic Split Reporting
    print("\n--- Chronological Split Diagnostics ---")
    for name, split_meta in metadata['splits'].items():
        print(f"Split [{name.upper()}]: {split_meta.num_windows} windows, "
              f"timestamps [{split_meta.start_timestamp} -> {split_meta.end_timestamp}], "
              f"total label entries: {split_meta.num_total_label_rows}")
        print(f"  Positive label counts: {split_meta.class_positive_counts}")
        if split_meta.unsupported_classes:
            print(f"  [NOTE] Unsupported classes (0 positives in this split): {split_meta.unsupported_classes}")

    train_pos_weights = metadata['train_pos_weights']
    print(f"\nDerived Training Class Weights (Eq. 6):")
    for fc, w in zip(metadata['fault_classes'], train_pos_weights.tolist()):
        print(f"  Class '{fc}': weight = {w:.2f}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # Randomize batch order within training split
        collate_fn=collate_holistic_rca
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_holistic_rca
    )

    # 2. Initialize Model Architecture
    entities = metadata['entities']
    fault_classes = metadata['fault_classes']
    feature_counts = metadata['feature_counts']

    model = HolisticRCA(
        node_types=['service'],
        feature_counts=feature_counts,
        num_time_buckets=window_size,
        embed_dim=embed_dim,
        entity_out_dim=entity_out_dim,
        num_fault_classes=len(fault_classes),
        fault_classes=fault_classes
    )
    model.to(device)

    # 3. Optimizer and Weighted BCE Loss Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = WeightedBCELoss(pos_weights={'service': train_pos_weights.to(device)})

    config = {
        'node_types': ['service'],
        'feature_counts': feature_counts,
        'num_time_buckets': window_size,
        'embed_dim': embed_dim,
        'entity_out_dim': entity_out_dim,
        'num_fault_classes': len(fault_classes),
        'fault_classes': fault_classes,
        'lr': lr,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'window_size': window_size,
        'seed': seed
    }

    history_records = []
    best_val_loss = float('inf')
    val_meta = metadata['splits']['val']
    can_validate_generalization = val_meta.has_positive_examples

    print("\n--- Starting Model Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_epoch_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            traces = batch['traces'].to(device)
            metrics = batch['metrics'].to(device)
            logs = batch['logs'].to(device)
            targets = batch['targets'].to(device)
            edge_index = batch['edge_index'].to(device)
            entity_indices = batch['entity_indices'].to(device)

            preds = model(
                traces=traces,
                metrics=metrics,
                logs=logs,
                entity_feature_indices=entity_indices,
                edge_index=edge_index
            )

            # Verify outputs are finite
            if torch.isnan(preds['service']).any() or torch.isinf(preds['service']).any():
                raise RuntimeError(f"Epoch {epoch}: Non-finite predictions detected!")

            loss = criterion(preds, {'service': targets})
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"Epoch {epoch}: Non-finite loss encountered!")

            loss.backward()

            # Verify gradients are finite
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        raise RuntimeError(f"Epoch {epoch}: Non-finite gradient in {name}!")

            optimizer.step()
            train_epoch_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_epoch_loss / max(1, train_batches)

        # Validation evaluation
        model.eval()
        val_epoch_loss = 0.0
        val_batches = 0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                traces = batch['traces'].to(device)
                metrics = batch['metrics'].to(device)
                logs = batch['logs'].to(device)
                targets = batch['targets'].to(device)
                edge_index = batch['edge_index'].to(device)
                entity_indices = batch['entity_indices'].to(device)

                preds = model(
                    traces=traces,
                    metrics=metrics,
                    logs=logs,
                    entity_feature_indices=entity_indices,
                    edge_index=edge_index
                )

                v_loss = criterion(preds, {'service': targets})
                val_epoch_loss += v_loss.item()
                val_batches += 1

                # Squeeze or flatten batch and entities for metric evaluation
                probs = torch.sigmoid(preds['service']).cpu().numpy()
                targs = targets.cpu().numpy()
                all_val_preds.append(probs.reshape(-1, len(fault_classes)))
                all_val_targets.append(targs.reshape(-1, len(fault_classes)))

        avg_val_loss = val_epoch_loss / max(1, val_batches)

        val_metrics = {}
        if can_validate_generalization and all_val_targets:
            y_true_val = np.concatenate(all_val_targets, axis=0)
            y_pred_val = np.concatenate(all_val_preds, axis=0)
            val_metrics = evaluate_fault_classification(y_true_val, y_pred_val)

        record = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_metrics': val_metrics
        }
        history_records.append(record)

        # Logging
        mif1_str = f", Val MiF1: {val_metrics.get('MiF1', 0.0):.4f}" if val_metrics else ""
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{mif1_str}")

        # Checkpointing
        latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
        save_checkpoint(
            checkpoint_path=latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            config=config,
            metadata=metadata,
            is_best_validated=False
        )

        # Save best checkpoint only when valid validation is available
        if can_validate_generalization and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
            save_checkpoint(
                checkpoint_path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                config=config,
                metadata=metadata,
                is_best_validated=True
            )

    # Save complete history and metadata
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    save_training_history(history_path, history_records, metadata)
    print(f"\nTraining completed successfully! Artifacts saved to: {checkpoint_dir}")

    return {
        'model': model,
        'config': config,
        'metadata': metadata,
        'final_train_loss': avg_train_loss,
        'final_val_loss': avg_val_loss,
        'history': history_records
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train HolisticRCA on Cloud-Native Observability Data")
    parser.add_argument("--data_dir", type=str, default="data_samples/dataset_c", help="Path to Dataset C folder")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Path to save checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size of temporal windows")
    parser.add_argument("--window_size", type=int, default=1, help="Number of time buckets per window")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_holistic_rca(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        window_size=args.window_size,
        seed=args.seed
    )
