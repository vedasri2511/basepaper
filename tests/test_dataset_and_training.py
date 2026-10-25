import sys
import os
import shutil
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import load_dataset_c_splits, collate_holistic_rca, HolisticRCADataset
from src.data.label_encoder import DEFAULT_SERVICE_FAULT_CLASSES
from src.models.holistic_rca import HolisticRCA
from src.models.losses import WeightedBCELoss
from src.models.checkpoint import save_checkpoint, load_checkpoint
from src.train import set_seed, train_holistic_rca

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_samples', 'dataset_c'))


def test_real_dataset_loader_and_alignment():
    """
    Validates:
    1. Real dataset loading of Dataset C CSV files.
    2. Alignment of traces, metrics, logs, RERG topology, and targets.
    3. Stable canonical ordering of 14 entities.
    4. Exact tensor shapes per window: (window_size, n_features) and (14, 3).
    """
    train_ds, val_ds, test_ds, meta = load_dataset_c_splits(
        data_dir=DATA_DIR,
        time_granularity=60,
        window_size=1,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )

    assert len(train_ds) == 36
    assert len(val_ds) == 12
    assert len(test_ds) == 12
    assert len(train_ds) + len(val_ds) + len(test_ds) == 60

    # Check item 0
    item = train_ds[0]
    assert item['trace'].shape == (1, 84)
    assert item['metric'].shape == (1, 98)
    assert item['log'].shape == (1, 84)
    assert item['target'].shape == (14, 3)
    assert item['edge_index'].dim() == 2
    assert item['entity_indices'].shape == (14, 19)
    assert isinstance(item['timestamp'], (int, np.integer))
    assert int(item['timestamp']) > 0

    # Check entities
    assert len(meta['entities']) == 14
    assert meta['entities'] == train_ds.entities
    print("test_real_dataset_loader_and_alignment passed.")


def test_leakage_aware_chronological_splits():
    """
    Validates:
    1. Splits are strictly non-overlapping and chronological.
    2. Train timestamps < Val timestamps < Test timestamps.
    3. No window appears in more than one split.
    4. Training-only class weights: val/test labels are not leaked.
    """
    train_ds, val_ds, test_ds, meta = load_dataset_c_splits(
        data_dir=DATA_DIR,
        time_granularity=60,
        window_size=1,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )

    train_ts = set(train_ds.timestamps)
    val_ts = set(val_ds.timestamps)
    test_ts = set(test_ds.timestamps)

    # No overlap
    assert len(train_ts.intersection(val_ts)) == 0, "Train and Val splits overlap!"
    assert len(val_ts.intersection(test_ts)) == 0, "Val and Test splits overlap!"
    assert len(train_ts.intersection(test_ts)) == 0, "Train and Test splits overlap!"

    # Strict chronological ordering
    assert max(train_ds.timestamps) < min(val_ds.timestamps)
    assert max(val_ds.timestamps) < min(test_ds.timestamps)

    # Training class weights computed strictly from train split
    train_weights = meta['train_pos_weights']
    assert train_weights.shape == (3,)
    # In train split (36 windows = 504 entity entries, exactly 6 cpu_load positives)
    # p_cpu_load = (504 - 6) / 6 = 498 / 6 = 83.0
    assert abs(train_weights[0].item() - 83.0) < 1e-4, f"Expected 83.0, got {train_weights[0].item()}"
    # Unsupported classes default to 1.0 safely
    assert abs(train_weights[1].item() - 1.0) < 1e-4
    assert abs(train_weights[2].item() - 1.0) < 1e-4
    print("test_leakage_aware_chronological_splits passed.")


def test_missing_positive_class_detection():
    """
    Validates that the split metadata explicitly identifies unsupported classes
    and splits lacking positive examples.
    """
    _, _, _, meta = load_dataset_c_splits(
        data_dir=DATA_DIR,
        time_granularity=60,
        window_size=1,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )

    train_meta = meta['splits']['train']
    val_meta = meta['splits']['val']
    test_meta = meta['splits']['test']

    # Train has cpu_load positives (6), but not network_delay or network_loss
    assert train_meta.class_positive_counts['cpu_load'] == 6
    assert 'network_delay' in train_meta.unsupported_classes
    assert 'network_loss' in train_meta.unsupported_classes

    # Val has cpu_load positives (4)
    assert val_meta.class_positive_counts['cpu_load'] == 4

    # Test has 0 positives for ALL classes (12 post-fault recovery windows)
    assert test_meta.class_positive_counts['cpu_load'] == 0
    assert test_meta.has_positive_examples is False
    assert set(test_meta.unsupported_classes) == {'cpu_load', 'network_delay', 'network_loss'}
    print("test_missing_positive_class_detection passed.")


def test_collate_and_batched_forward_pass():
    """
    Validates DataLoader batching, collate function, and forward pass returning (B, 14, 3) logits.
    """
    train_ds, _, _, meta = load_dataset_c_splits(
        data_dir=DATA_DIR,
        time_granularity=60,
        window_size=1
    )

    loader = DataLoader(train_ds, batch_size=4, shuffle=False, collate_fn=collate_holistic_rca)
    batch = next(iter(loader))

    assert batch['traces'].shape == (4, 1, 84)
    assert batch['metrics'].shape == (4, 1, 98)
    assert batch['logs'].shape == (4, 1, 84)
    assert batch['targets'].shape == (4, 14, 3)

    model = HolisticRCA(
        node_types=['service'],
        feature_counts=meta['feature_counts'],
        num_time_buckets=1,
        embed_dim=128,
        entity_out_dim=256,
        num_fault_classes=3,
        fault_classes=DEFAULT_SERVICE_FAULT_CLASSES
    )

    preds = model(
        traces=batch['traces'],
        metrics=batch['metrics'],
        logs=batch['logs'],
        entity_feature_indices=batch['entity_indices'],
        edge_index=batch['edge_index']
    )

    assert 'service' in preds
    assert preds['service'].shape == (4, 14, 3)
    assert not torch.isnan(preds['service']).any()
    print("test_collate_and_batched_forward_pass passed.")


def test_minimal_training_step_and_gradient_flow():
    """
    Validates:
    1. Single optimization step on a real Dataset C batch.
    2. Finite loss calculation using WeightedBCELoss.
    3. Active gradient propagation into all model parameters.
    """
    train_ds, _, _, meta = load_dataset_c_splits(data_dir=DATA_DIR, window_size=1)
    loader = DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=collate_holistic_rca)
    batch = next(iter(loader))

    model = HolisticRCA(
        node_types=['service'],
        feature_counts=meta['feature_counts'],
        num_time_buckets=1,
        embed_dim=64,
        entity_out_dim=128,
        num_fault_classes=3
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = WeightedBCELoss(pos_weights={'service': meta['train_pos_weights']})

    model.train()
    optimizer.zero_grad()

    preds = model(
        traces=batch['traces'],
        metrics=batch['metrics'],
        logs=batch['logs'],
        entity_feature_indices=batch['entity_indices'],
        edge_index=batch['edge_index']
    )
    loss = criterion(preds, {'service': batch['targets']})
    assert not torch.isnan(loss) and not torch.isinf(loss)
    assert loss.item() > 0.0

    loss.backward()

    # Verify non-zero gradients in linear map, entity integrator, gat, and classifier
    assert model.trace_rep.linear_map.weight.grad is not None
    assert model.trace_rep.linear_map.weight.grad.norm().item() > 0
    assert model.entity_integrators['service'].linear_out.weight.grad is not None
    assert model.entity_integrators['service'].linear_out.weight.grad.norm().item() > 0
    assert model.gat.lin_l.weight.grad is not None
    assert model.gat.lin_l.weight.grad.norm().item() > 0
    assert model.classifiers['service'][1].weight.grad is not None
    assert model.classifiers['service'][1].weight.grad.norm().item() > 0

    optimizer.step()
    print("test_minimal_training_step_and_gradient_flow passed.")


def test_validation_mode_does_not_modify_parameters():
    """
    Validates that model.eval() and torch.no_grad() do NOT alter weights.
    """
    _, val_ds, _, meta = load_dataset_c_splits(data_dir=DATA_DIR, window_size=1)
    loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_holistic_rca)
    batch = next(iter(loader))

    model = HolisticRCA(
        node_types=['service'],
        feature_counts=meta['feature_counts'],
        num_time_buckets=1,
        embed_dim=64,
        entity_out_dim=128,
        num_fault_classes=3
    )

    # Record initial weights
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

    model.eval()
    with torch.no_grad():
        preds = model(
            traces=batch['traces'],
            metrics=batch['metrics'],
            logs=batch['logs'],
            entity_feature_indices=batch['entity_indices'],
            edge_index=batch['edge_index']
        )

    for k, v in model.state_dict().items():
        assert torch.equal(v, initial_weights[k]), f"Weight {k} was altered during validation!"

    print("test_validation_mode_does_not_modify_parameters passed.")


def test_checkpoint_save_load_consistency():
    """
    Validates that saving a checkpoint and loading it restores identical model state
    and produces deterministic, identical outputs.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        train_ds, _, _, meta = load_dataset_c_splits(data_dir=DATA_DIR, window_size=1)
        loader = DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=collate_holistic_rca)
        batch = next(iter(loader))

        config = {
            'node_types': ['service'],
            'feature_counts': meta['feature_counts'],
            'num_time_buckets': 1,
            'embed_dim': 64,
            'entity_out_dim': 128,
            'num_fault_classes': 3,
            'fault_classes': DEFAULT_SERVICE_FAULT_CLASSES
        }

        model = HolisticRCA(**config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        ckpt_path = os.path.join(temp_dir, "test_ckpt.pt")
        save_checkpoint(
            checkpoint_path=ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=1,
            train_loss=1.234,
            val_loss=0.567,
            config=config,
            metadata=meta,
            is_best_validated=False
        )
        assert os.path.exists(ckpt_path)

        # Original forward output
        model.eval()
        with torch.no_grad():
            orig_out = model(
                traces=batch['traces'],
                metrics=batch['metrics'],
                logs=batch['logs'],
                entity_feature_indices=batch['entity_indices'],
                edge_index=batch['edge_index']
            )['service']

        # Reconstruct model from checkpoint
        reloaded_model, ckpt_dict = load_checkpoint(ckpt_path)
        reloaded_model.eval()
        with torch.no_grad():
            reloaded_out = reloaded_model(
                traces=batch['traces'],
                metrics=batch['metrics'],
                logs=batch['logs'],
                entity_feature_indices=batch['entity_indices'],
                edge_index=batch['edge_index']
            )['service']

        assert torch.equal(orig_out, reloaded_out), "Reloaded model output differs from original!"
        assert ckpt_dict['epoch'] == 1
        assert ckpt_dict['train_loss'] == 1.234
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("test_checkpoint_save_load_consistency passed.")


def test_reproducibility_with_seed():
    """
    Validates that setting fixed random seeds produces identical model weights
    and data initialization.
    """
    set_seed(12345)
    m1 = HolisticRCA(node_types=['service'], feature_counts={'service': 19}, num_time_buckets=1, embed_dim=32, entity_out_dim=64)

    set_seed(12345)
    m2 = HolisticRCA(node_types=['service'], feature_counts={'service': 19}, num_time_buckets=1, embed_dim=32, entity_out_dim=64)

    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert k1 == k2
        assert torch.equal(v1, v2), f"Parameter {k1} differs under same seed!"

    print("test_reproducibility_with_seed passed.")


class TestDatasetAndTraining(unittest.TestCase):

    def test_real_dataset_loader_and_alignment(self):
        test_real_dataset_loader_and_alignment()

    def test_leakage_aware_chronological_splits(self):
        test_leakage_aware_chronological_splits()

    def test_missing_positive_class_detection(self):
        test_missing_positive_class_detection()

    def test_collate_and_batched_forward_pass(self):
        test_collate_and_batched_forward_pass()

    def test_minimal_training_step_and_gradient_flow(self):
        test_minimal_training_step_and_gradient_flow()

    def test_validation_mode_does_not_modify_parameters(self):
        test_validation_mode_does_not_modify_parameters()

    def test_checkpoint_save_load_consistency(self):
        test_checkpoint_save_load_consistency()

    def test_reproducibility_with_seed(self):
        test_reproducibility_with_seed()


if __name__ == '__main__':
    unittest.main()
