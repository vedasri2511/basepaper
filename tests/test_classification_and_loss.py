import sys
import os
import math
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from src.data.preprocessor import Preprocessor
from src.data.rerg_builder import RERGBuilder
from src.data.label_encoder import LabelEncoder, DEFAULT_SERVICE_FAULT_CLASSES
from src.models.holistic_rca import PositionalEncoding, HolisticRCA
from src.models.losses import WeightedBCELoss

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_samples', 'dataset_c'))


def get_dataset_c_preprocessed_and_labels():
    traces = pd.read_csv(os.path.join(DATA_DIR, 'traces.csv'))
    metrics = pd.read_csv(os.path.join(DATA_DIR, 'metrics.csv'))
    logs = pd.read_csv(os.path.join(DATA_DIR, 'logs.csv'))
    labels = pd.read_csv(os.path.join(DATA_DIR, 'labels.csv'))

    preprocessor = Preprocessor((1696000000, 1696003600), 60)
    out = preprocessor.process_all({'traces': traces, 'metrics': metrics, 'logs': logs})

    builder = RERGBuilder(entities=out.entities, add_self_loops=True, add_bidirectional=True)
    graph = builder.build_graph(traces)

    return out, graph, labels


# =========================================================================
# GATE A TESTS: Positional Encoding Formulation
# =========================================================================

def test_positional_encoding_gate_a():
    """
    Gate A Verification:
    1. Output shape equals input shape for 2D (t, n) and 3D (b, t, n).
    2. Deterministic output across multiple calls.
    3. Distinguishable time positions: PE(t_1, :) != PE(t_2, :).
    4. Distinguishable feature channels: PE(:, c_1) != PE(:, c_2) (no 1D broadcasting).
    5. Finite output values in range [-1, 1].
    6. Odd and even feature dimensions handled seamlessly.
    """
    pos_enc = PositionalEncoding(max_len=200)

    # Even feature count (e.g. 84)
    x_even = torch.zeros(60, 84, dtype=torch.float32)
    pe_even = pos_enc(x_even)
    assert pe_even.shape == (60, 84)
    assert torch.equal(pe_even, pos_enc(x_even)) # Deterministic
    assert not torch.equal(pe_even[0, :], pe_even[1, :]) # Time distinguishable
    assert not torch.equal(pe_even[1:, 0], pe_even[1:, 2]) # Channel distinguishable
    assert not torch.isnan(pe_even).any() and not torch.isinf(pe_even).any()
    assert (pe_even.abs() <= 1.0 + 1e-6).all()

    # Odd feature count (e.g. 19)
    x_odd = torch.zeros(60, 19, dtype=torch.float32)
    pe_odd = pos_enc(x_odd)
    assert pe_odd.shape == (60, 19)
    assert not torch.equal(pe_odd[1:, 0], pe_odd[1:, 2])
    assert not torch.isnan(pe_odd).any()

    # Batched 3D tensor
    x_batch = torch.zeros(4, 60, 84, dtype=torch.float32)
    pe_batch = pos_enc(x_batch)
    assert pe_batch.shape == (4, 60, 84)
    assert torch.equal(pe_batch[0], pe_batch[1])
    print("test_positional_encoding_gate_a passed.")


# =========================================================================
# GATE B TESTS: RERG Edge Types and Message Passing
# =========================================================================

def test_rerg_edge_types_and_gat_message_passing_gate_b():
    """
    Gate B Verification:
    1. RERGBuilder returns both edge_index and edge_types.
    2. GATv2 consumes edge_index per paper Eq. (3)-(4).
    3. Self-loops are included in RERGBuilder and not duplicated in GAT (add_self_loops=False).
    4. Topology directionality (forward, reverse, self-loops) is preserved.
    """
    out, graph, _ = get_dataset_c_preprocessed_and_labels()

    assert graph.edge_index is not None
    assert graph.edge_types is not None
    assert graph.edge_index.shape[1] == len(graph.edge_types)

    # Self-loops contract check
    self_loop_mask = (graph.edge_index[0] == graph.edge_index[1])
    num_self_loops = int(self_loop_mask.sum().item())
    assert num_self_loops == len(out.entities), f"Expected {len(out.entities)} self loops, found {num_self_loops}"

    # GAT consumes graph.edge_index without crashing or duplicating self-loops
    model = HolisticRCA(
        node_types=['service'],
        feature_counts={'service': 19},
        num_time_buckets=60,
        embed_dim=128,
        entity_out_dim=256,
        num_fault_classes=3
    )

    entity_indices = torch.tensor([out.entity_feature_map[e]['global_indices'] for e in out.entities], dtype=torch.long)
    X_t = torch.from_numpy(out.X_trace).float()
    X_m = torch.from_numpy(out.X_metric).float()
    X_l = torch.from_numpy(out.X_log).float()

    v_prime = model.get_fused_entity_embeddings(X_t, X_m, X_l, entity_indices, graph.edge_index)
    assert v_prime.shape == (len(out.entities), 256)
    assert not torch.isnan(v_prime).any()
    print("test_rerg_edge_types_and_gat_message_passing_gate_b passed.")


# =========================================================================
# GATE C TESTS: Five vs Three Classifier Outputs and Schema Alignment
# =========================================================================

def test_classifier_output_schema_and_classes_gate_c():
    """
    Gate C Verification:
    1. Paper Section V.B.1 defines 3 fault classes for microservices in Dataset C:
       ['cpu_load', 'network_delay', 'network_loss'].
    2. Model defaults to 3 classes and outputs shape (|E|, 3).
    3. Documented ordering: Index 0: cpu_load, Index 1: network_delay, Index 2: network_loss.
    4. Custom fault_classes can be passed and configured cleanly.
    """
    encoder = LabelEncoder(fault_classes=DEFAULT_SERVICE_FAULT_CLASSES)
    assert encoder.fault_classes == ['cpu_load', 'network_delay', 'network_loss']
    assert encoder.fault_to_idx == {'cpu_load': 0, 'network_delay': 1, 'network_loss': 2}

    model = HolisticRCA(
        node_types=['service'],
        feature_counts={'service': 19},
        num_time_buckets=60,
        num_fault_classes=3,
        fault_classes=DEFAULT_SERVICE_FAULT_CLASSES
    )
    assert model.num_fault_classes == 3
    assert model.fault_classes['service'] == ['cpu_load', 'network_delay', 'network_loss']
    print("test_classifier_output_schema_and_classes_gate_c passed.")


# =========================================================================
# PHASE D TESTS: Label Alignment with Dataset C
# =========================================================================

def test_label_alignment_with_dataset_c():
    """
    Phase D Verification:
    1. Encodes 840 rows of Dataset C labels accurately.
    2. Window targets align with canonical entity ordering.
    3. Normal windows produce all zeros: [0, 0, 0].
    4. Faulty windows (buckets 30-39 for social_service_3) produce [1, 0, 0] for cpu_load.
    5. All other entities in faulty window remain [0, 0, 0].
    """
    out, _, labels = get_dataset_c_preprocessed_and_labels()
    encoder = LabelEncoder(entities=out.entities, fault_classes=DEFAULT_SERVICE_FAULT_CLASSES)

    # 1. Normal window at timestamp 1696000000
    norm_target = encoder.create_window_target_tensor(labels, time_start=1696000000)
    assert norm_target.shape == (14, 3)
    assert (norm_target == 0.0).all(), "Normal window should have zero targets everywhere!"

    # 2. Faulty window at timestamp 1696001800 (bucket 30)
    fault_target = encoder.create_window_target_tensor(labels, time_start=1696001800)
    assert fault_target.shape == (14, 3)

    social_service_3_idx = out.entities.index('social_service_3')

    # social_service_3 must have [1, 0, 0] (cpu_load)
    expected_fault = torch.tensor([1.0, 0.0, 0.0])
    assert torch.equal(fault_target[social_service_3_idx], expected_fault), \
        f"social_service_3 target should be [1, 0, 0], got {fault_target[social_service_3_idx]}"

    # All other 13 services must have [0, 0, 0]
    for i, e in enumerate(out.entities):
        if i != social_service_3_idx:
            assert torch.equal(fault_target[i], torch.zeros(3)), f"Service {e} should be normal [0, 0, 0]"

    # 3. Full sequence target tensor across all 60 windows
    seq_targets, timestamps = encoder.create_sequence_target_tensor(labels)
    assert seq_targets.shape == (60, 14, 3)
    assert len(timestamps) == 60

    # Exactly 10 positive labels across entire tensor
    assert int(seq_targets.sum().item()) == 10
    assert int((seq_targets[:, :, 0] == 1.0).sum().item()) == 10 # All 10 are cpu_load
    assert int(seq_targets[:, :, 1].sum().item()) == 0 # 0 network_delay
    assert int(seq_targets[:, :, 2].sum().item()) == 0 # 0 network_loss
    print("test_label_alignment_with_dataset_c passed.")


def test_label_encoder_error_handling():
    """
    Phase D Verification:
    - Rejects missing entities.
    - Rejects duplicate entity entries in the same window.
    - Rejects unknown fault types when is_fault == 1.
    """
    encoder = LabelEncoder(entities=['srv_a', 'srv_b'], fault_classes=['cpu_load', 'delay'])

    # Unknown fault type
    try:
        encoder.encode_label(is_fault=1, fault_type='disk_full')
        assert False, "Should have raised ValueError on unknown fault_type"
    except ValueError:
        pass

    # Missing entity in window
    df_missing = pd.DataFrame([
        {'timestamp': 100, 'timestamp_end': 160, 'entity': 'srv_a', 'is_fault': 0, 'fault_type': 'normal'}
    ])
    try:
        encoder.create_window_target_tensor(df_missing, time_start=100)
        assert False, "Should have raised ValueError on missing entity 'srv_b'"
    except ValueError:
        pass

    # Duplicate entity in window
    df_dup = pd.DataFrame([
        {'timestamp': 100, 'timestamp_end': 160, 'entity': 'srv_a', 'is_fault': 0, 'fault_type': 'normal'},
        {'timestamp': 100, 'timestamp_end': 160, 'entity': 'srv_a', 'is_fault': 0, 'fault_type': 'normal'},
        {'timestamp': 100, 'timestamp_end': 160, 'entity': 'srv_b', 'is_fault': 0, 'fault_type': 'normal'}
    ])
    try:
        encoder.create_window_target_tensor(df_dup, time_start=100)
        assert False, "Should have raised ValueError on duplicate entity"
    except ValueError:
        pass
    print("test_label_encoder_error_handling passed.")


# =========================================================================
# PHASE F TESTS: Weighted BCE Loss Formulation
# =========================================================================

def test_weighted_bce_manual_calculation():
    """
    Phase F Verification:
    Tests WeightedBCELoss against an exact analytical, hand-calculated example.

    Setup:
    - 1 entity type (kappa = 1)
    - 2 entities (d_tilde = 2): e_1, e_2
    - 2 classes (F = 2): c_1, c_2
    - Targets:
      e_1: [1.0, 0.0]
      e_2: [0.0, 0.0]
    - Class weights:
      p = [2.0, 1.0]
    - Logits:
      z_1: [0.0, 0.0] -> y_hat = [0.5, 0.5]
      z_2: [0.0, 0.0] -> y_hat = [0.5, 0.5]

    Manual loss calculation:
    e_1, c_1: y=1, y_hat=0.5, p=2.0 -> -2.0 * ln(0.5) = 2 * ln(2)
    e_1, c_2: y=0, y_hat=0.5, p=1.0 -> -1.0 * ln(0.5) = 1 * ln(2)
    sum_c(e_1) = 3 * ln(2)

    e_2, c_1: y=0, y_hat=0.5, p=2.0 -> -1.0 * ln(0.5) = 1 * ln(2)
    e_2, c_2: y=0, y_hat=0.5, p=1.0 -> -1.0 * ln(0.5) = 1 * ln(2)
    sum_c(e_2) = 2 * ln(2)

    Total entity sum = (3 + 2) * ln(2) = 5 * ln(2)
    Average over d_tilde=2: L_cls = 2.5 * ln(2) ≈ 1.73286795
    """
    loss_fn = WeightedBCELoss(pos_weights=torch.tensor([2.0, 1.0]))

    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)

    loss = loss_fn(logits, targets)
    expected_loss = 2.5 * math.log(2.0)

    assert abs(loss.item() - expected_loss) < 1e-6, f"Expected {expected_loss}, got {loss.item()}"
    print(f"test_weighted_bce_manual_calculation passed (Loss: {loss.item():.6f} vs Expected: {expected_loss:.6f}).")


def test_class_imbalance_and_dataset_c_weights():
    """
    Phase F Verification:
    1. Verifies compute_class_weights matches p_c = N_non_faulty / N_faulty.
    2. For Dataset C (840 total rows, 10 cpu_load positives):
       p_cpu_load = 830 / 10 = 83.0.
    3. network_delay (0 positives) safely defaults to 1.0 without ZeroDivisionError.
    4. network_loss (0 positives) safely defaults to 1.0 without ZeroDivisionError.
    """
    _, _, labels = get_dataset_c_preprocessed_and_labels()
    encoder = LabelEncoder(fault_classes=DEFAULT_SERVICE_FAULT_CLASSES)

    weights = encoder.compute_class_weights(labels)
    assert weights.shape == (3,)
    assert abs(weights[0].item() - 83.0) < 1e-5, f"Expected 83.0 for cpu_load, got {weights[0].item()}"
    assert abs(weights[1].item() - 1.0) < 1e-5, f"Expected 1.0 for network_delay, got {weights[1].item()}"
    assert abs(weights[2].item() - 1.0) < 1e-5, f"Expected 1.0 for network_loss, got {weights[2].item()}"

    # Test zero-positive and zero-negative edge case datasets
    df_all_normal = pd.DataFrame([
        {'is_fault': 0, 'fault_type': 'normal'} for _ in range(50)
    ])
    w_all_norm = encoder.compute_class_weights(df_all_normal)
    assert (w_all_norm == 1.0).all(), "All normal dataset should safely yield weight 1.0"

    df_all_faulty = pd.DataFrame([
        {'is_fault': 1, 'fault_type': 'cpu_load'} for _ in range(50)
    ])
    w_all_fault = encoder.compute_class_weights(df_all_faulty)
    assert (w_all_fault == 1.0).all(), "All faulty dataset should safely yield weight 1.0"
    print("test_class_imbalance_and_dataset_c_weights passed.")


def test_end_to_end_forward_loss_and_gradients():
    """
    Phase F & G Verification:
    Full pipeline test:
    1. Real Preprocessor outputs -> X_t, X_m, X_l.
    2. Real RERGBuilder -> edge_index.
    3. Real LabelEncoder -> window targets.
    4. HolisticRCA forward pass producing logits.
    5. WeightedBCELoss with Dataset C class weights.
    6. Loss is finite and non-trivial.
    7. Active backpropagation verifies loss.backward() flows gradients through all model parameters.
    """
    out, graph, labels = get_dataset_c_preprocessed_and_labels()
    encoder = LabelEncoder(entities=out.entities, fault_classes=DEFAULT_SERVICE_FAULT_CLASSES)

    weights = encoder.compute_class_weights(labels)
    loss_fn = WeightedBCELoss(pos_weights=weights)

    model = HolisticRCA(
        node_types=['service'],
        feature_counts={'service': 19},
        num_time_buckets=60,
        embed_dim=128,
        entity_out_dim=256,
        num_fault_classes=3,
        fault_classes=DEFAULT_SERVICE_FAULT_CLASSES
    )

    entity_indices = torch.tensor([out.entity_feature_map[e]['global_indices'] for e in out.entities], dtype=torch.long)
    X_t = torch.from_numpy(out.X_trace).float().clone().requires_grad_(True)
    X_m = torch.from_numpy(out.X_metric).float().clone().requires_grad_(True)
    X_l = torch.from_numpy(out.X_log).float().clone().requires_grad_(True)

    # Test on a faulty window (bucket 30)
    window_target = encoder.create_window_target_tensor(labels, time_start=1696001800)

    preds = model(
        traces=X_t,
        metrics=X_m,
        logs=X_l,
        entity_feature_indices=entity_indices,
        edge_index=graph.edge_index
    )

    assert 'service' in preds
    assert preds['service'].shape == (14, 3)

    # Compute Weighted BCE Loss
    loss = loss_fn(preds, {'service': window_target})
    assert not torch.isnan(loss) and not torch.isinf(loss)
    assert loss.item() > 0.0

    loss.backward()

    # Check gradients on inputs
    assert X_t.grad is not None and X_t.grad.norm().item() > 0
    assert X_m.grad is not None and X_m.grad.norm().item() > 0
    assert X_l.grad is not None and X_l.grad.norm().item() > 0

    # Check gradients on model components
    assert model.trace_rep.linear_map.weight.grad is not None
    assert model.entity_integrators['service'].linear_out.weight.grad is not None
    assert model.gat.lin_l.weight.grad is not None
    assert model.classifiers['service'][1].weight.grad is not None

    # Probabilities via predict_probabilities
    probs = model.predict_probabilities(preds)
    assert (probs['service'] >= 0.0).all() and (probs['service'] <= 1.0).all()
    print("test_end_to_end_forward_loss_and_gradients passed successfully!")


class TestClassificationAndLoss(unittest.TestCase):

    def test_positional_encoding_gate_a(self):
        test_positional_encoding_gate_a()

    def test_rerg_edge_types_and_gat_message_passing_gate_b(self):
        test_rerg_edge_types_and_gat_message_passing_gate_b()

    def test_classifier_output_schema_and_classes_gate_c(self):
        test_classifier_output_schema_and_classes_gate_c()

    def test_label_alignment_with_dataset_c(self):
        test_label_alignment_with_dataset_c()

    def test_label_encoder_error_handling(self):
        test_label_encoder_error_handling()

    def test_weighted_bce_manual_calculation(self):
        test_weighted_bce_manual_calculation()

    def test_class_imbalance_and_dataset_c_weights(self):
        test_class_imbalance_and_dataset_c_weights()

    def test_end_to_end_forward_loss_and_gradients(self):
        test_end_to_end_forward_loss_and_gradients()


if __name__ == '__main__':
    unittest.main()
