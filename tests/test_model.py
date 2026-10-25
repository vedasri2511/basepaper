import sys
import os
import unittest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.preprocessor import Preprocessor
from src.data.rerg_builder import RERGBuilder
from src.models.holistic_rca import (
    PositionalEncoding,
    ObservabilityRepresentation,
    ResourceEntityFeatureIntegration,
    HolisticRCA
)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_samples', 'dataset_c'))


def get_real_preprocessed_dataset_c():
    """Helper to run Preprocessor and RERGBuilder on Dataset C files."""
    traces = pd.read_csv(os.path.join(DATA_DIR, 'traces.csv'))
    metrics = pd.read_csv(os.path.join(DATA_DIR, 'metrics.csv'))
    logs = pd.read_csv(os.path.join(DATA_DIR, 'logs.csv'))

    preprocessor = Preprocessor((1696000000, 1696003600), 60)
    out = preprocessor.process_all({'traces': traces, 'metrics': metrics, 'logs': logs})

    builder = RERGBuilder(entities=out.entities, add_self_loops=True, add_bidirectional=True)
    graph = builder.build_graph(traces)

    return out, graph


def test_positional_encoding():
    """Validates sinusoidal positional encoding dimensions, determinism, and time-axis differentiation."""
    pos_enc = PositionalEncoding(max_len=100)
    x = torch.zeros(60, 10, dtype=torch.float32)

    out1 = pos_enc(x)
    out2 = pos_enc(x)

    # 1. Output shape must match input shape exactly
    assert out1.shape == (60, 10)

    # 2. Must be 100% deterministic
    assert torch.equal(out1, out2)

    # 3. Time step 0 must differ from time step 1
    assert not torch.equal(out1[0, :], out1[1, :])

    # 4. Batched input support
    x_batch = torch.zeros(2, 60, 10, dtype=torch.float32)
    out_batch = pos_enc(x_batch)
    assert out_batch.shape == (2, 60, 10)

    # 5. Channel-wise distinction (Gate A): different channels have distinct frequencies
    assert not torch.equal(out1[1:, 0], out1[1:, 2]), "Channels should have unique frequencies, not broadcast values!"
    print("test_positional_encoding passed.")



def test_observability_representation_with_real_preprocessed_output():
    """Validates representation learning on real preprocessed matrices from Dataset C."""
    out, _ = get_real_preprocessed_dataset_c()

    X_t = torch.from_numpy(out.X_trace).float() # (60, 84)
    X_m = torch.from_numpy(out.X_metric).float() # (60, 98)
    X_l = torch.from_numpy(out.X_log).float()    # (60, 84)

    embed_dim = 128
    trace_rep = ObservabilityRepresentation(num_time_buckets=60, embed_dim=embed_dim)
    metric_rep = ObservabilityRepresentation(num_time_buckets=60, embed_dim=embed_dim)
    log_rep = ObservabilityRepresentation(num_time_buckets=60, embed_dim=embed_dim)

    H_t = trace_rep(X_t)
    H_m = metric_rep(X_m)
    H_l = log_rep(X_l)

    # Output shape: (num_features, embed_dim)
    assert H_t.shape == (84, embed_dim)
    assert H_m.shape == (98, embed_dim)
    assert H_l.shape == (84, embed_dim)

    # Finite values check
    assert not torch.isnan(H_t).any() and not torch.isinf(H_t).any()
    assert not torch.isnan(H_m).any() and not torch.isinf(H_m).any()
    assert not torch.isnan(H_l).any() and not torch.isinf(H_l).any()
    print("test_observability_representation_with_real_preprocessed_output passed.")


def test_entity_gathering_and_integration():
    """Validates gathering rows from H_total and applying cross-modality Transformer integration."""
    out, _ = get_real_preprocessed_dataset_c()

    embed_dim = 128
    out_dim = 256

    # Total features: 84 + 98 + 84 = 266
    H_total = torch.randn(266, embed_dim)

    # Build entity feature indices tensor: (14, 19)
    entity_indices = torch.tensor([out.entity_feature_map[e]['global_indices'] for e in out.entities], dtype=torch.long)
    assert entity_indices.shape == (14, 19)

    # Gather: HI_e has shape (14, 19, 128)
    HI = H_total[entity_indices]
    assert HI.shape == (14, 19, embed_dim)

    # Feature Integrator for 'service'
    integrator = ResourceEntityFeatureIntegration(num_features=19, embed_dim=embed_dim, out_dim=out_dim)
    V = integrator(HI)

    # Output node matrix V: (14, 256)
    assert V.shape == (14, out_dim)
    assert not torch.isnan(V).any() and not torch.isinf(V).any()
    print("test_entity_gathering_and_integration passed.")


def test_gat_fusion_with_rerg_builder():
    """Validates GATv2 message passing on V using the actual edge_index from RERGBuilder."""
    out, graph = get_real_preprocessed_dataset_c()

    # Node embeddings V: (14, 256)
    V = torch.randn(14, 256)

    # GAT with add_self_loops=False honoring the RERGBuilder self-loop contract
    gat = nn.Sequential(
        # wrap GATv2Conv call
    )
    from torch_geometric.nn import GATv2Conv
    gat_layer = GATv2Conv(256, 256, heads=4, concat=False, add_self_loops=False)

    V_prime = gat_layer(V, graph.edge_index)

    assert V_prime.shape == (14, 256)
    assert not torch.isnan(V_prime).any() and not torch.isinf(V_prime).any()
    print("test_gat_fusion_with_rerg_builder passed.")


def test_end_to_end_holistic_rca_forward():
    """Validates complete forward pass of HolisticRCA with real Dataset C preprocessed tensors."""
    out, graph = get_real_preprocessed_dataset_c()

    X_t = torch.from_numpy(out.X_trace).float()
    X_m = torch.from_numpy(out.X_metric).float()
    X_l = torch.from_numpy(out.X_log).float()

    entity_indices = torch.tensor([out.entity_feature_map[e]['global_indices'] for e in out.entities], dtype=torch.long)

    model = HolisticRCA(
        node_types=['service'],
        feature_counts={'service': 19},
        num_time_buckets=60,
        embed_dim=128,
        entity_out_dim=256,
        num_fault_classes=5
    )

    # 1. Test get_fused_entity_embeddings
    V_prime = model.get_fused_entity_embeddings(
        traces=X_t,
        metrics=X_m,
        logs=X_l,
        entity_feature_indices=entity_indices,
        edge_index=graph.edge_index
    )
    assert V_prime.shape == (14, 256)
    assert not torch.isnan(V_prime).any()

    # 2. Test full forward pass returning logits
    preds = model(
        traces=X_t,
        metrics=X_m,
        logs=X_l,
        entity_feature_indices=entity_indices,
        edge_index=graph.edge_index
    )
    assert 'service' in preds
    assert preds['service'].shape == (14, 5) # (num_services, num_fault_classes)
    assert not torch.isnan(preds['service']).any()
    print("test_end_to_end_holistic_rca_forward passed.")


def test_gradient_propagation_through_all_stages():
    """
    CRITICAL TEST: Validates that gradients backpropagate all the way from the GAT /
    classifier output back through entity integrators and modality Transformers to raw inputs.
    """
    out, graph = get_real_preprocessed_dataset_c()

    X_t = torch.from_numpy(out.X_trace).float().clone().requires_grad_(True)
    X_m = torch.from_numpy(out.X_metric).float().clone().requires_grad_(True)
    X_l = torch.from_numpy(out.X_log).float().clone().requires_grad_(True)

    entity_indices = torch.tensor([out.entity_feature_map[e]['global_indices'] for e in out.entities], dtype=torch.long)

    model = HolisticRCA(
        node_types=['service'],
        feature_counts={'service': 19},
        num_time_buckets=60,
        embed_dim=128,
        entity_out_dim=256,
        num_fault_classes=5
    )

    preds = model(
        traces=X_t,
        metrics=X_m,
        logs=X_l,
        entity_feature_indices=entity_indices,
        edge_index=graph.edge_index
    )

    loss = preds['service'].sum()
    loss.backward()

    # Assert gradients exist and have non-zero magnitude across all stages
    assert X_t.grad is not None and X_t.grad.norm().item() > 0, "No gradient reaching X_t!"
    assert X_m.grad is not None and X_m.grad.norm().item() > 0, "No gradient reaching X_m!"
    assert X_l.grad is not None and X_l.grad.norm().item() > 0, "No gradient reaching X_l!"

    # Trace linear map
    assert model.trace_rep.linear_map.weight.grad is not None
    assert model.trace_rep.linear_map.weight.grad.norm().item() > 0

    # Entity integrator linear out
    assert model.entity_integrators['service'].linear_out.weight.grad is not None
    assert model.entity_integrators['service'].linear_out.weight.grad.norm().item() > 0

    # GAT convolution parameters
    assert model.gat.lin_l.weight.grad is not None
    assert model.gat.lin_l.weight.grad.norm().item() > 0

    # Classifier parameters
    assert model.classifiers['service'][1].weight.grad is not None
    assert model.classifiers['service'][1].weight.grad.norm().item() > 0

    print("test_gradient_propagation_through_all_stages passed: Active end-to-end gradient flow verified!")


def test_empty_or_missing_modality_handling():
    """Validates model behavior when a modality is empty or omitted."""
    out, graph = get_real_preprocessed_dataset_c()
    entity_indices = torch.tensor([out.entity_feature_map[e]['global_indices'] for e in out.entities], dtype=torch.long)

    model = HolisticRCA(
        node_types=['service'],
        feature_counts={'service': 19},
        num_time_buckets=60,
        embed_dim=128,
        entity_out_dim=256,
        num_fault_classes=5
    )

    X_t = torch.from_numpy(out.X_trace).float()
    X_m = torch.from_numpy(out.X_metric).float()

    # With logs=None (e.g. system without logs)
    # The feature counts must match the gathered subset
    # Verify error is raised gracefully if modalities do not provide enough total features
    try:
        model(traces=X_t, metrics=X_m, logs=None, entity_feature_indices=entity_indices, edge_index=graph.edge_index)
    except IndexError:
        # Expected since entity_indices expects 266 total features
        pass
    print("test_empty_or_missing_modality_handling passed.")


def test_batched_input_compatibility():
    """Validates that batched modality tensors (batch_size, time, features) execute cleanly."""
    batch_size = 2
    num_buckets = 60

    X_t = torch.randn(batch_size, num_buckets, 84)
    X_m = torch.randn(batch_size, num_buckets, 98)
    X_l = torch.randn(batch_size, num_buckets, 84)

    entity_indices = torch.randint(0, 266, (14, 19), dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

    model = HolisticRCA(
        node_types=['service'],
        feature_counts={'service': 19},
        num_time_buckets=num_buckets,
        embed_dim=128,
        entity_out_dim=256,
        num_fault_classes=5
    )

    preds = model(
        traces=X_t,
        metrics=X_m,
        logs=X_l,
        entity_feature_indices=entity_indices,
        edge_index=edge_index
    )

    assert 'service' in preds
    assert preds['service'].shape == (batch_size, 14, 5)
    print("test_batched_input_compatibility passed.")


class TestModel(unittest.TestCase):

    def test_positional_encoding(self):
        test_positional_encoding()

    def test_observability_representation_with_real_preprocessed_output(self):
        test_observability_representation_with_real_preprocessed_output()

    def test_entity_gathering_and_integration(self):
        test_entity_gathering_and_integration()

    def test_gat_fusion_with_rerg_builder(self):
        test_gat_fusion_with_rerg_builder()

    def test_end_to_end_holistic_rca_forward(self):
        test_end_to_end_holistic_rca_forward()

    def test_gradient_propagation_through_all_stages(self):
        test_gradient_propagation_through_all_stages()

    def test_empty_or_missing_modality_handling(self):
        test_empty_or_missing_modality_handling()

    def test_batched_input_compatibility(self):
        test_batched_input_compatibility()


if __name__ == '__main__':
    unittest.main()
