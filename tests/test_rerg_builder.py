import sys
import os
import unittest
import pandas as pd
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.preprocessor import Preprocessor
from src.data.rerg_builder import (
    RERGBuilder,
    RERGGraph,
    REL_CALL_FORWARD,
    REL_CALL_REVERSE,
    REL_SELF_LOOP,
    REL_DEPLOYMENT
)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_samples', 'dataset_c'))


def test_node_count_and_stable_indices():
    """Validates node count, natural sorting, and bijective node index mapping."""
    entities = [f"social_service_{i}" for i in range(14)]
    # Shuffle to ensure natural sort key behaves deterministically
    shuffled_entities = list(reversed(entities))
    builder = RERGBuilder(entities=shuffled_entities)

    assert builder.num_nodes == 14
    assert builder.entities[0] == "social_service_0"
    assert builder.entities[1] == "social_service_1"
    assert builder.entities[2] == "social_service_2"
    assert builder.entities[10] == "social_service_10"
    assert builder.entities[13] == "social_service_13"

    for i, ent in enumerate(builder.entities):
        assert builder.node_to_idx[ent] == i
    print("test_node_count_and_stable_indices passed.")


def test_known_span_relationship_extraction():
    """Validates extraction of forward and reverse call edges from known parent-child spans."""
    spans = pd.DataFrame([
        {'span_id': 's1', 'parent_span_id': None, 'service_name': 'serviceA', 'start_time': 100},
        {'span_id': 's2', 'parent_span_id': 's1', 'service_name': 'serviceB', 'start_time': 105},
        {'span_id': 's3', 'parent_span_id': 's2', 'service_name': 'serviceC', 'start_time': 110},
    ])
    builder = RERGBuilder(entities=['serviceA', 'serviceB', 'serviceC'], add_self_loops=False, add_bidirectional=True)
    graph = builder.build_graph(spans)

    # Expected edges: (A->B, B->A), (B->C, C->B) = 4 edges
    assert graph.num_edges == 4
    edges = set(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))

    idxA, idxB, idxC = builder.node_to_idx['serviceA'], builder.node_to_idx['serviceB'], builder.node_to_idx['serviceC']
    assert (idxA, idxB) in edges
    assert (idxB, idxA) in edges
    assert (idxB, idxC) in edges
    assert (idxC, idxB) in edges
    print("test_known_span_relationship_extraction passed.")


def test_missing_parent_root_and_self_spans():
    """Validates robust handling of root spans, missing/dangling parent IDs, and intraservice self-calls."""
    spans = pd.DataFrame([
        {'span_id': 'root_1', 'parent_span_id': None, 'service_name': 'serviceA', 'start_time': 100},
        {'span_id': 'root_2', 'parent_span_id': '', 'service_name': 'serviceB', 'start_time': 101},
        {'span_id': 'dangling', 'parent_span_id': 'non_existent_id', 'service_name': 'serviceC', 'start_time': 102},
        {'span_id': 'self_child', 'parent_span_id': 'root_1', 'service_name': 'serviceA', 'start_time': 103}, # self-call
        {'span_id': 'valid_call', 'parent_span_id': 'root_1', 'service_name': 'serviceB', 'start_time': 104}, # A -> B
        {'span_id': 'dup_call', 'parent_span_id': 'root_1', 'service_name': 'serviceB', 'start_time': 105},   # A -> B duplicate
    ])
    builder = RERGBuilder(entities=['serviceA', 'serviceB', 'serviceC'], add_self_loops=False, add_bidirectional=True)
    graph = builder.build_graph(spans)

    stats = graph.stats
    assert stats['root_spans'] == 2
    assert stats['unresolved_parent_spans'] == 1
    assert stats['self_calls'] == 1
    assert stats['valid_inter_service_calls'] == 2 # 2 instances of A -> B
    assert stats['active_call_relations_count'] == 1 # 1 unique pair

    # Graph should have exactly 2 edges (A->B and B->A), deduplicated
    assert graph.num_edges == 2
    idxA, idxB = builder.node_to_idx['serviceA'], builder.node_to_idx['serviceB']
    edges = set(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))
    assert (idxA, idxB) in edges
    assert (idxB, idxA) in edges
    print("test_missing_parent_root_and_self_spans passed.")


def test_cross_window_parent_child_resolution():
    """
    REGRESSION TEST: Parent span starts in Window 0, child span starts in Window 1.
    Verifies that the child span in Window 1 successfully resolves its parent service
    using global span resolution without marking it as unresolved.
    """
    spans = pd.DataFrame([
        # Window 0 [0, 60): Parent span in serviceA
        {'span_id': 'parent_span_w0', 'parent_span_id': None, 'service_name': 'serviceA', 'start_time': 50},
        # Window 1 [60, 120): Child span in serviceB referencing parent from Window 0
        {'span_id': 'child_span_w1', 'parent_span_id': 'parent_span_w0', 'service_name': 'serviceB', 'start_time': 70},
    ])
    builder = RERGBuilder(entities=['serviceA', 'serviceB'], add_self_loops=False, add_bidirectional=True)
    intervals = [(0, 60), (60, 120)]
    graphs = builder.build_dynamic_graphs(spans, intervals)

    # Window 0: contains root span parent_span_w0 (no inter-service call executed yet)
    g_w0 = graphs[0]
    assert g_w0.stats['root_spans'] == 1
    assert g_w0.num_edges == 0

    # Window 1: contains child_span_w1. Must resolve parent_span_w0 -> serviceA across boundary
    g_w1 = graphs[1]
    assert g_w1.stats['unresolved_parent_spans'] == 0, "Cross-window parent was incorrectly marked unresolved!"
    assert g_w1.stats['valid_inter_service_calls'] == 1
    assert g_w1.num_edges == 2 # serviceA <-> serviceB

    idxA, idxB = builder.node_to_idx['serviceA'], builder.node_to_idx['serviceB']
    edges_w1 = set(zip(g_w1.edge_index[0].tolist(), g_w1.edge_index[1].tolist()))
    assert (idxA, idxB) in edges_w1
    assert (idxB, idxA) in edges_w1
    print("test_cross_window_parent_child_resolution passed.")


def test_dynamic_topology_variation_across_windows():
    """
    REGRESSION TEST: Controlled synthetic test where service-call relationships
    dynamically vary across time windows. Confirms the graph structure changes
    genuinely without inventing fabricated edges.
    """
    spans = pd.DataFrame([
        # Window 0 [0, 60): Only A -> B
        {'span_id': 's1_w0', 'parent_span_id': None, 'service_name': 'serviceA', 'start_time': 10},
        {'span_id': 's2_w0', 'parent_span_id': 's1_w0', 'service_name': 'serviceB', 'start_time': 15},

        # Window 1 [60, 120): A -> B AND B -> C
        {'span_id': 's1_w1', 'parent_span_id': None, 'service_name': 'serviceA', 'start_time': 70},
        {'span_id': 's2_w1', 'parent_span_id': 's1_w1', 'service_name': 'serviceB', 'start_time': 75},
        {'span_id': 's3_w1', 'parent_span_id': 's2_w1', 'service_name': 'serviceC', 'start_time': 80},

        # Window 2 [120, 180): Only C -> D
        {'span_id': 's1_w2', 'parent_span_id': None, 'service_name': 'serviceC', 'start_time': 130},
        {'span_id': 's2_w2', 'parent_span_id': 's1_w2', 'service_name': 'serviceD', 'start_time': 135},
    ])
    entities = ['serviceA', 'serviceB', 'serviceC', 'serviceD']
    builder = RERGBuilder(entities=entities, add_self_loops=True, add_bidirectional=True)
    intervals = [(0, 60), (60, 120), (120, 180)]
    graphs = builder.build_dynamic_graphs(spans, intervals)

    # Window 0: A<->B (2 edges) + 4 self-loops = 6 edges
    assert graphs[0].num_edges == 6
    edges_0 = set(zip(graphs[0].edge_index[0].tolist(), graphs[0].edge_index[1].tolist()))
    iA, iB, iC, iD = builder.node_to_idx['serviceA'], builder.node_to_idx['serviceB'], builder.node_to_idx['serviceC'], builder.node_to_idx['serviceD']
    assert (iA, iB) in edges_0 and (iB, iA) in edges_0
    assert (iB, iC) not in edges_0

    # Window 1: A<->B, B<->C (4 edges) + 4 self-loops = 8 edges
    assert graphs[1].num_edges == 8
    edges_1 = set(zip(graphs[1].edge_index[0].tolist(), graphs[1].edge_index[1].tolist()))
    assert (iA, iB) in edges_1 and (iB, iC) in edges_1

    # Window 2: C<->D (2 edges) + 4 self-loops = 6 edges
    assert graphs[2].num_edges == 6
    edges_2 = set(zip(graphs[2].edge_index[0].tolist(), graphs[2].edge_index[1].tolist()))
    assert (iC, iD) in edges_2 and (iD, iC) in edges_2
    assert (iA, iB) not in edges_2
    print("test_dynamic_topology_variation_across_windows passed.")


def test_deployment_relation_safety_and_stats():
    """
    Validates that deployment relations safely integrate when entities exist,
    and explicitly record unregistered entities in stats without crashing or generating bad indices.
    """
    entities = ['serviceA', 'serviceB']
    builder = RERGBuilder(entities=entities, add_self_loops=False, add_bidirectional=True)

    deployment_relations = [
        ('serviceA', 'serviceB'),            # Valid
        ('serviceA', 'unregistered_pod_1'),  # Missing child
        ('unregistered_node_0', 'serviceB')  # Missing parent
    ]
    graph = builder.build_graph(pd.DataFrame(), deployment_relations=deployment_relations)

    assert graph.stats['unregistered_deployment_entities'] == 2
    # Only valid pair (A <-> B) should produce 2 edges
    assert graph.num_edges == 2
    edges = set(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))
    assert (0, 1) in edges and (1, 0) in edges
    print("test_deployment_relation_safety_and_stats passed.")


def test_empty_input_handling():
    """Validates behavior on empty input DataFrames or None."""
    entities = ['serviceA', 'serviceB', 'serviceC']
    builder_with_loops = RERGBuilder(entities=entities, add_self_loops=True)
    builder_no_loops = RERGBuilder(entities=entities, add_self_loops=False)

    g_empty_loops = builder_with_loops.build_graph(pd.DataFrame())
    assert g_empty_loops.num_edges == 3 # 3 self-loops
    assert g_empty_loops.edge_index.shape == (2, 3)

    g_empty_no_loops = builder_no_loops.build_graph(pd.DataFrame())
    assert g_empty_no_loops.num_edges == 0
    assert g_empty_no_loops.edge_index.shape == (2, 0)
    print("test_empty_input_handling passed.")


def test_tensor_dimensions_and_valid_indices():
    """Validates edge tensor dimensions, dtypes, and bounds on Dataset C traces."""
    traces = pd.read_csv(os.path.join(DATA_DIR, 'traces.csv'))
    entities = [f"social_service_{i}" for i in range(14)]
    builder = RERGBuilder(entities=entities, add_self_loops=True, add_bidirectional=True)
    graph = builder.build_graph(traces)

    assert isinstance(graph.edge_index, torch.Tensor)
    assert graph.edge_index.dtype == torch.long
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] == graph.num_edges

    assert isinstance(graph.edge_types, torch.Tensor)
    assert graph.edge_types.dtype == torch.long
    assert graph.edge_types.shape[0] == graph.num_edges

    # Ensure all edge indices are valid node indices
    assert graph.edge_index.min().item() >= 0
    assert graph.edge_index.max().item() < 14
    print("test_tensor_dimensions_and_valid_indices passed.")


def test_dataset_c_actual_topology_no_fabricated_edges():
    """
    Confirms exact trace topology derived from Dataset C without fabricating edges.
    Traces contain only:
    social_service_0 -> social_service_1 -> social_service_2 -> social_service_3 -> social_service_4.
    Services 5 to 13 must NOT have fabricated trace edges.
    """
    traces = pd.read_csv(os.path.join(DATA_DIR, 'traces.csv'))
    entities = [f"social_service_{i}" for i in range(14)]
    builder = RERGBuilder(entities=entities, add_self_loops=False, add_bidirectional=True)
    graph = builder.build_graph(traces)

    # 4 call pairs * 2 (bidirectional) = 8 edges
    assert graph.num_edges == 8

    edges = set(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))
    expected_pairs = {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)}
    assert edges == expected_pairs

    # Verify no trace edges touch services 5 to 13
    connected_nodes = set(graph.edge_index[0].tolist()).union(set(graph.edge_index[1].tolist()))
    assert connected_nodes == {0, 1, 2, 3, 4}
    for isolated_idx in range(5, 14):
        assert isolated_idx not in connected_nodes
    print("test_dataset_c_actual_topology_no_fabricated_edges passed.")


def test_timestamp_window_alignment_and_dynamic_behavior():
    """Validates dynamic graph generation per 60-second time interval matching preprocessor & labels.csv."""
    traces = pd.read_csv(os.path.join(DATA_DIR, 'traces.csv'))
    labels = pd.read_csv(os.path.join(DATA_DIR, 'labels.csv'))
    entities = [f"social_service_{i}" for i in range(14)]

    preprocessor = Preprocessor((1696000000, 1696003600), 60)
    intervals = preprocessor.partition_time()
    assert len(intervals) == 60

    builder = RERGBuilder(entities=entities, add_self_loops=True, add_bidirectional=True)
    dynamic_graphs = builder.build_dynamic_graphs(traces, intervals)

    assert len(dynamic_graphs) == 60
    label_ts = sorted(labels['timestamp'].unique())

    for idx, g in enumerate(dynamic_graphs):
        expected_t_start, expected_t_end = intervals[idx]
        assert g.timestamp_start == expected_t_start
        assert g.timestamp_end == expected_t_end
        assert g.timestamp_start == label_ts[idx]
        assert g.num_nodes == 14
        # Every bucket in Dataset C has traces for the 4 call pairs + 14 self loops = 22 edges
        assert g.num_edges == 22
    print("test_timestamp_window_alignment_and_dynamic_behavior passed.")


def test_deterministic_graph_construction():
    """Validates that repeated construction yields identical tensors."""
    traces = pd.read_csv(os.path.join(DATA_DIR, 'traces.csv'))
    entities = [f"social_service_{i}" for i in range(14)]
    builder = RERGBuilder(entities=entities, add_self_loops=True, add_bidirectional=True)

    g1 = builder.build_graph(traces)
    g2 = builder.build_graph(traces)

    assert torch.equal(g1.edge_index, g2.edge_index)
    assert torch.equal(g1.edge_types, g2.edge_types)
    print("test_deterministic_graph_construction passed.")


def test_compatibility_with_preprocessing_mappings():
    """Validates exact index alignment between preprocessor entity mapping and RERG graph node indices."""
    traces = pd.read_csv(os.path.join(DATA_DIR, 'traces.csv'))
    metrics = pd.read_csv(os.path.join(DATA_DIR, 'metrics.csv'))
    logs = pd.read_csv(os.path.join(DATA_DIR, 'logs.csv'))

    preprocessor = Preprocessor((1696000000, 1696003600), 60)
    prep_out = preprocessor.process_all({'traces': traces, 'metrics': metrics, 'logs': logs})

    builder = RERGBuilder(entities=prep_out.entities, add_self_loops=True, add_bidirectional=True)
    graph = builder.build_graph(traces)

    assert graph.num_nodes == len(prep_out.entities) == 14
    for ent in prep_out.entities:
        prep_idx = prep_out.entity_feature_map[ent]['entity_index']
        graph_idx = builder.node_to_idx[ent]
        assert prep_idx == graph_idx, f"Index mismatch for {ent}: preprocessor={prep_idx}, graph={graph_idx}"
    print("test_compatibility_with_preprocessing_mappings passed.")


class TestRERGBuilder(unittest.TestCase):

    def test_node_count_and_stable_indices(self):
        test_node_count_and_stable_indices()

    def test_known_span_relationship_extraction(self):
        test_known_span_relationship_extraction()

    def test_missing_parent_root_and_self_spans(self):
        test_missing_parent_root_and_self_spans()

    def test_cross_window_parent_child_resolution(self):
        test_cross_window_parent_child_resolution()

    def test_dynamic_topology_variation_across_windows(self):
        test_dynamic_topology_variation_across_windows()

    def test_deployment_relation_safety_and_stats(self):
        test_deployment_relation_safety_and_stats()

    def test_empty_input_handling(self):
        test_empty_input_handling()

    def test_tensor_dimensions_and_valid_indices(self):
        test_tensor_dimensions_and_valid_indices()

    def test_dataset_c_actual_topology_no_fabricated_edges(self):
        test_dataset_c_actual_topology_no_fabricated_edges()

    def test_timestamp_window_alignment_and_dynamic_behavior(self):
        test_timestamp_window_alignment_and_dynamic_behavior()

    def test_deterministic_graph_construction(self):
        test_deterministic_graph_construction()

    def test_compatibility_with_preprocessing_mappings(self):
        test_compatibility_with_preprocessing_mappings()


if __name__ == '__main__':
    unittest.main()
