import re
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch


def natural_sort_key(s: str) -> List[Any]:
    """Helper for natural alphanumeric sorting (e.g. social_service_2 before social_service_10)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


# Relation type identifiers conforming to Section IV.D:
# - REL_CALL_FORWARD: Trace invocation where caller service invokes callee service.
# - REL_CALL_REVERSE: Reverse edge added pursuant to paper Section IV.D ("for each relation,
#                     we add two edges between the relevant resource entities, with one
#                     resource entity pointing to the other and its reverse edge").
# - REL_SELF_LOOP: Self-loop (u -> u) ensuring node self-representation is preserved during GAT aggregation.
# - REL_DEPLOYMENT: Deployment topology relation (Service <-> Pod, Pod <-> Node) for heterogeneous graphs.
REL_CALL_FORWARD = 0
REL_CALL_REVERSE = 1
REL_SELF_LOOP = 2
REL_DEPLOYMENT = 3


@dataclass
class RERGGraph:
    """
    Standardized container for a Resource Entity Relation Graph (RERG).
    Conforms to Section IV.D of HolisticRCA (Yongqi Han et al., IEEE TSC 2024).
    Compatible with PyTorch Geometric / GAT downstream consumption.

    Ownership Contract:
    When add_self_loops=True (default), self-loops are explicitly added to edge_index.
    The downstream GAT layer should be initialized with add_self_loops=False to prevent
    duplicate self-loop aggregation.
    """
    edge_index: torch.Tensor             # Shape: (2, num_edges), dtype torch.long
    edge_types: torch.Tensor             # Shape: (num_edges,), dtype torch.long
    num_nodes: int                       # Total entity count |E|
    node_names: List[str]                # Canonical entity identifiers ordered by node index
    node_to_idx: Dict[str, int]          # Bijective mapping from entity name to node index
    edge_type_names: List[str] = field(default_factory=lambda: [
        "call_forward", "call_reverse", "self_loop", "deployment"
    ])
    timestamp_start: Optional[int] = None
    timestamp_end: Optional[int] = None
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1] if self.edge_index is not None else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'node_names': self.node_names,
            'node_to_idx': self.node_to_idx,
            'edge_index': self.edge_index.tolist() if self.edge_index is not None else [],
            'edge_types': self.edge_types.tolist() if self.edge_types is not None else [],
            'timestamp_start': self.timestamp_start,
            'timestamp_end': self.timestamp_end,
            'stats': self.stats
        }


class RERGBuilder:
    """
    Constructs the Resource Entity Relation Graph (RERG) according to Section IV.D.

    Paper Specification:
    - Nodes: Resource entities (services, pods, nodes).
    - Edges: Relations derived from system topology.
      1) Call relations among services derived from traces.
      2) System deployment relations (e.g. service to pods, pod to node).
    - Directionality: "For each relation, we add two edges between the relevant
      resource entities, with one resource entity pointing to the other and its reverse edge."
    """
    def __init__(
        self,
        entities: List[str],
        add_self_loops: bool = True,
        add_bidirectional: bool = True
    ):
        """
        entities: List of entity names (must agree with PreprocessedOutput.entities)
        add_self_loops: Whether to include (u, u) self-edges for GAT attention retention
        add_bidirectional: Whether to add reverse edge for every call relation (Paper Eq. 3-4)
        """
        self.entities = sorted(entities, key=natural_sort_key)
        self.node_to_idx = {ent: i for i, ent in enumerate(self.entities)}
        self.num_nodes = len(self.entities)
        self.add_self_loops = add_self_loops
        self.add_bidirectional = add_bidirectional

    def extract_call_relations(
        self,
        spans_df: pd.DataFrame,
        time_window: Optional[Tuple[int, int]] = None,
        global_span_lookup: Optional[Dict[str, str]] = None
    ) -> Tuple[List[Tuple[str, str, int]], Dict[str, int]]:
        """
        Derives service invocation pairs from trace span parent-child linkages.

        Window Boundary Resolution:
        - If global_span_lookup is provided (or built from the full spans_df), any child span
          executing in [t_start, t_end) can resolve its parent service even if the parent
          span started in an earlier time window.

        Returns:
            call_relations: List of (caller_service, callee_service, invocation_count)
            stats: Dictionary tracking extraction diagnostics
        """
        stats = {
            'total_spans_in_window': 0,
            'root_spans': 0,
            'unresolved_parent_spans': 0,
            'self_calls': 0,
            'valid_inter_service_calls': 0,
            'unregistered_entities': 0
        }

        if spans_df is None or spans_df.empty:
            return [], stats

        # Build global span lookup if not provided
        if global_span_lookup is None:
            global_span_lookup = {}
            for _, row in spans_df.iterrows():
                s_id = str(row['span_id']).strip()
                global_span_lookup[s_id] = str(row['service_name']).strip()

        # Filter active spans by time window if specified
        if time_window is not None:
            t_start, t_end = time_window
            spans = spans_df[
                (spans_df['start_time'] >= t_start) &
                (spans_df['start_time'] < t_end)
            ].copy()
        else:
            spans = spans_df.copy()

        stats['total_spans_in_window'] = len(spans)
        if spans.empty:
            return [], stats

        call_counts = Counter()
        for _, row in spans.iterrows():
            child_svc = str(row['service_name']).strip()
            parent_id = row.get('parent_span_id', None)

            # Case 1: Root span (no parent invocation)
            if pd.isna(parent_id) or str(parent_id).strip() == '' or str(parent_id).lower() == 'nan':
                stats['root_spans'] += 1
                continue

            parent_id_str = str(parent_id).strip()

            # Case 2: Parent span cannot be resolved across the global trace dataset
            if parent_id_str not in global_span_lookup:
                stats['unresolved_parent_spans'] += 1
                continue

            parent_svc = global_span_lookup[parent_id_str]

            # Case 3: Intraservice span (internal function invocation within the same service)
            if parent_svc == child_svc:
                stats['self_calls'] += 1
                continue

            # Case 4: Entity not in registered entity list
            if parent_svc not in self.node_to_idx or child_svc not in self.node_to_idx:
                stats['unregistered_entities'] += 1
                continue

            # Case 5: Valid inter-service invocation
            call_counts[(parent_svc, child_svc)] += 1
            stats['valid_inter_service_calls'] += 1

        call_relations = [
            (caller, callee, count)
            for (caller, callee), count in sorted(call_counts.items())
        ]
        return call_relations, stats

    def build_graph(
        self,
        spans_df: pd.DataFrame,
        time_window: Optional[Tuple[int, int]] = None,
        deployment_relations: Optional[List[Tuple[str, str]]] = None,
        global_span_lookup: Optional[Dict[str, str]] = None
    ) -> RERGGraph:
        """
        Constructs the RERG graph for the given trace spans and optional deployment topology.

        Args:
            spans_df: DataFrame with ['span_id', 'parent_span_id', 'service_name', 'start_time']
            time_window: Optional (t_start, t_end) for dynamic windowing
            deployment_relations: Optional list of (service, pod) or (pod, node) tuples
            global_span_lookup: Optional precomputed span_id -> service_name dictionary

        Returns:
            RERGGraph instance with edge_index tensor of shape (2, num_edges)
        """
        call_relations, stats = self.extract_call_relations(
            spans_df,
            time_window=time_window,
            global_span_lookup=global_span_lookup
        )

        edges: List[Tuple[int, int]] = []
        edge_types: List[int] = []
        seen_edges: Set[Tuple[int, int, int]] = set()

        def add_edge(src_idx: int, dst_idx: int, rel_type: int):
            key = (src_idx, dst_idx, rel_type)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append((src_idx, dst_idx))
                edge_types.append(rel_type)

        # 1. Trace Invocation Edges (Caller -> Callee and Callee -> Caller)
        for caller, callee, count in call_relations:
            u = self.node_to_idx[caller]
            v = self.node_to_idx[callee]

            # Forward call edge
            add_edge(u, v, REL_CALL_FORWARD)

            # Paper Section IV.D: Reverse edge for global information propagation
            if self.add_bidirectional:
                add_edge(v, u, REL_CALL_REVERSE)

        # 2. Deployment Topology Edges (e.g. Service <-> Pod, Pod <-> Node)
        stats['unregistered_deployment_entities'] = 0
        if deployment_relations:
            for parent_ent, child_ent in deployment_relations:
                p_in = parent_ent in self.node_to_idx
                c_in = child_ent in self.node_to_idx
                if p_in and c_in:
                    u = self.node_to_idx[parent_ent]
                    v = self.node_to_idx[child_ent]
                    add_edge(u, v, REL_DEPLOYMENT)
                    if self.add_bidirectional:
                        add_edge(v, u, REL_DEPLOYMENT)
                else:
                    stats['unregistered_deployment_entities'] += 1

        # 3. Self-loops (u -> u) for GAT representation preservation
        if self.add_self_loops:
            for i in range(self.num_nodes):
                add_edge(i, i, REL_SELF_LOOP)

        # Convert to PyTorch tensors
        if edges:
            src_list = [e[0] for e in edges]
            dst_list = [e[1] for e in edges]
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_types_tensor = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_types_tensor = torch.empty((0,), dtype=torch.long)

        t_start = time_window[0] if time_window else None
        t_end = time_window[1] if time_window else None

        stats['num_edges_total'] = edge_index.shape[1]
        stats['num_nodes_total'] = self.num_nodes
        stats['active_call_relations_count'] = len(call_relations)

        return RERGGraph(
            edge_index=edge_index,
            edge_types=edge_types_tensor,
            num_nodes=self.num_nodes,
            node_names=list(self.entities),
            node_to_idx=dict(self.node_to_idx),
            timestamp_start=t_start,
            timestamp_end=t_end,
            stats=stats
        )

    def build_dynamic_graphs(
        self,
        spans_df: pd.DataFrame,
        time_intervals: List[Tuple[int, int]]
    ) -> List[RERGGraph]:
        """
        Constructs a sequence of RERG graphs across multiple time intervals (e.g. 60-second buckets).
        Precomputes the global span lookup once to ensure parent spans across window boundaries
        are resolved with O(N) efficiency and zero dropped references.
        """
        global_span_lookup = {}
        if spans_df is not None and not spans_df.empty:
            for _, row in spans_df.iterrows():
                s_id = str(row['span_id']).strip()
                global_span_lookup[s_id] = str(row['service_name']).strip()

        graphs = []
        for interval in time_intervals:
            g = self.build_graph(
                spans_df,
                time_window=interval,
                global_span_lookup=global_span_lookup
            )
            graphs.append(g)
        return graphs
