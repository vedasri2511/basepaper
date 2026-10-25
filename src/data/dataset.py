"""
Dataset Loader and Temporal Splitter for HolisticRCA.
Aligned with paper: "Holistic Root Cause Analysis for Failures in Cloud-Native Systems Through Observability Data"
(Yongqi Han et al., IEEE TSC 2024).

This module implements:
1. HolisticRCADataset: Window-aligned PyTorch Dataset for observability data (traces, metrics, logs) and RERG graph.
2. Temporal split manager: Chronological, non-overlapping train/validation/test partitioning.
3. Strict data leakage prevention:
   - Log vocabulary fitted exclusively on training logs.
   - Class imbalance weights computed exclusively on training labels.
4. Diagnostic reporting of class distributions and unsupported fault types per split.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import os
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessor import Preprocessor, PreprocessedOutput, natural_sort_key
from src.data.rerg_builder import RERGBuilder, RERGGraph
from src.data.label_encoder import LabelEncoder, DEFAULT_SERVICE_FAULT_CLASSES


@dataclass
class SplitMetadata:
    """Diagnostic metadata for a chronological dataset split."""
    name: str
    start_timestamp: int
    end_timestamp: int
    num_windows: int
    num_total_label_rows: int
    class_positive_counts: Dict[str, int]
    class_negative_counts: Dict[str, int]
    unsupported_classes: List[str]
    has_positive_examples: bool

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'start_timestamp': self.start_timestamp,
            'end_timestamp': self.end_timestamp,
            'num_windows': self.num_windows,
            'num_total_label_rows': self.num_total_label_rows,
            'class_positive_counts': self.class_positive_counts,
            'class_negative_counts': self.class_negative_counts,
            'unsupported_classes': self.unsupported_classes,
            'has_positive_examples': self.has_positive_examples
        }



class HolisticRCADataset(Dataset):
    """
    PyTorch Dataset providing aligned (X_trace, X_metric, X_log, edge_index, targets)
    per analysis window.

    Temporal Alignment:
    - Each sample represents an observation window of `window_size` buckets (default: 1 bucket = 60s).
    - Window i covers timestamp interval [t_i, t_i + window_size * dt).
    - Output target is of shape (|E|, num_classes) corresponding to the window ground truth.
    """
    def __init__(
        self,
        X_trace: np.ndarray,      # (num_windows, window_size, n_trace)
        X_metric: np.ndarray,     # (num_windows, window_size, n_metric)
        X_log: np.ndarray,        # (num_windows, window_size, n_log)
        targets: torch.Tensor,    # (num_windows, num_entities, num_classes)
        edge_index: torch.Tensor, # (2, num_edges)
        entity_indices: torch.Tensor, # (num_entities, n_e)
        timestamps: List[int],
        entities: List[str],
        fault_classes: List[str],
        split_name: str = "train"
    ):
        assert len(X_trace) == len(X_metric) == len(X_log) == len(targets) == len(timestamps), \
            f"Dimension mismatch across windows: {len(X_trace)}, {len(X_metric)}, {len(X_log)}, {len(targets)}, {len(timestamps)}"

        self.X_trace = torch.from_numpy(X_trace).float()
        self.X_metric = torch.from_numpy(X_metric).float()
        self.X_log = torch.from_numpy(X_log).float()
        self.targets = targets.float()
        self.edge_index = edge_index.long()
        self.entity_indices = entity_indices.long()
        self.timestamps = timestamps
        self.entities = entities
        self.fault_classes = fault_classes
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            'trace': self.X_trace[idx],          # (window_size, n_trace)
            'metric': self.X_metric[idx],        # (window_size, n_metric)
            'log': self.X_log[idx],              # (window_size, n_log)
            'target': self.targets[idx],         # (num_entities, num_classes)
            'edge_index': self.edge_index,       # (2, num_edges)
            'entity_indices': self.entity_indices, # (num_entities, n_e)
            'timestamp': self.timestamps[idx],
            'split_name': self.split_name
        }


def collate_holistic_rca(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader: batches temporal windows while preserving
    shared graph topology and entity feature gathering indices.
    """
    traces = torch.stack([item['trace'] for item in batch], dim=0)   # (B, window_size, n_trace)
    metrics = torch.stack([item['metric'] for item in batch], dim=0) # (B, window_size, n_metric)
    logs = torch.stack([item['log'] for item in batch], dim=0)       # (B, window_size, n_log)
    targets = torch.stack([item['target'] for item in batch], dim=0) # (B, num_entities, num_classes)

    edge_index = batch[0]['edge_index']
    entity_indices = batch[0]['entity_indices']
    timestamps = [item['timestamp'] for item in batch]

    return {
        'traces': traces,
        'metrics': metrics,
        'logs': logs,
        'targets': targets,
        'edge_index': edge_index,
        'entity_indices': entity_indices,
        'timestamps': timestamps
    }


def load_dataset_c_splits(
    data_dir: str = "data_samples/dataset_c",
    time_granularity: int = 60,
    window_size: int = 1,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    fault_classes: Optional[List[str]] = None
) -> Tuple[HolisticRCADataset, HolisticRCADataset, HolisticRCADataset, Dict[str, Any]]:
    """
    Loads raw Dataset C CSV files, executes leakage-aware preprocessing,
    creates chronological non-overlapping splits, and computes training class weights.

    Leakage Prevention Guarantees:
    1. Splits are strictly chronological and non-overlapping.
    2. Log vocabulary is mined using Drain3/TF-IDF strictly from training logs.
    3. Positive class weights p_c are calculated strictly from training labels.

    Returns:
        (train_dataset, val_dataset, test_dataset, metadata_dict)
    """
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-4), \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    classes = fault_classes if fault_classes is not None else DEFAULT_SERVICE_FAULT_CLASSES

    # 1. Load raw files
    traces_path = os.path.join(data_dir, "traces.csv")
    metrics_path = os.path.join(data_dir, "metrics.csv")
    logs_path = os.path.join(data_dir, "logs.csv")
    labels_path = os.path.join(data_dir, "labels.csv")

    for p in [traces_path, metrics_path, logs_path, labels_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required dataset file not found: {p}")

    traces_df = pd.read_csv(traces_path)
    metrics_df = pd.read_csv(metrics_path)
    logs_df = pd.read_csv(logs_path)
    labels_df = pd.read_csv(labels_path)

    # Validate label columns
    required_label_cols = {'timestamp', 'timestamp_end', 'entity', 'is_fault', 'fault_type'}
    if not required_label_cols.issubset(labels_df.columns):
        raise ValueError(f"labels.csv missing required columns: {required_label_cols - set(labels_df.columns)}")

    # Determine full timeline from labels
    unique_timestamps = [int(ts) for ts in sorted(labels_df['timestamp'].unique())]
    num_total_buckets = len(unique_timestamps)
    if num_total_buckets == 0:
        raise ValueError("No timestamps found in labels.csv")

    t_min = int(unique_timestamps[0])
    t_max = int(labels_df['timestamp_end'].max())


    # 2. Compute chronological split indices
    n_train_windows = int(round(num_total_buckets * train_ratio))
    n_val_windows = int(round(num_total_buckets * val_ratio))
    n_test_windows = num_total_buckets - n_train_windows - n_val_windows

    if n_train_windows <= 0 or n_val_windows <= 0 or n_test_windows <= 0:
        raise ValueError(
            f"Invalid split configuration: train={n_train_windows}, val={n_val_windows}, test={n_test_windows}"
        )

    train_ts = unique_timestamps[:n_train_windows]
    val_ts = unique_timestamps[n_train_windows:n_train_windows + n_val_windows]
    test_ts = unique_timestamps[n_train_windows + n_val_windows:]

    train_end_time = train_ts[-1] + time_granularity
    val_end_time = val_ts[-1] + time_granularity

    # 3. Fit Preprocessor Log Vocabulary ONLY on Training Logs (Strict Leakage Prevention)
    train_logs_df = logs_df[logs_df['timestamp'] < train_end_time]

    preprocessor = Preprocessor(
        time_interval=(t_min, t_max),
        time_granularity=time_granularity
    )
    # Mine vocabulary strictly from training logs
    train_log_vocab = preprocessor.build_log_vocabulary(train_logs_df)

    # Process full interval using the training-fitted vocabulary
    preprocessed_out = preprocessor.process_all(
        raw_data={'traces': traces_df, 'metrics': metrics_df, 'logs': logs_df},
        log_vocab=train_log_vocab
    )

    entities = preprocessed_out.entities # canonical order of 14 services

    # 4. Build RERG Topology Graph
    rerg_builder = RERGBuilder(entities=entities, add_self_loops=True, add_bidirectional=True)
    graph = rerg_builder.build_graph(traces_df)

    # Entity gathering indices tensor (|E|, 19)
    entity_indices = torch.tensor(
        [preprocessed_out.entity_feature_map[e]['global_indices'] for e in entities],
        dtype=torch.long
    )

    # 5. Label Encoder & Target Tensor Construction
    label_encoder = LabelEncoder(entities=entities, fault_classes=classes)
    full_targets, target_timestamps = label_encoder.create_sequence_target_tensor(labels_df)

    assert target_timestamps == unique_timestamps, "Timestamp mismatch between labels and preprocessor!"

    # 6. Compute Positive Class Weights ONLY from Training Labels (Strict Leakage Prevention)
    train_labels_df = labels_df[labels_df['timestamp'].isin(train_ts)]
    val_labels_df = labels_df[labels_df['timestamp'].isin(val_ts)]
    test_labels_df = labels_df[labels_df['timestamp'].isin(test_ts)]

    train_pos_weights = label_encoder.compute_class_weights(train_labels_df)

    # 7. Reshape inputs into windows of size `window_size`
    # X_trace: (60, 84) -> (num_windows, window_size, 84)
    def create_window_arrays(X_full: np.ndarray, targets_full: torch.Tensor, timestamps_list: List[int]):
        X_windows = []
        targets_windows = []
        ts_windows = []

        for i in range(len(timestamps_list) - window_size + 1):
            idx = timestamps_list.index(timestamps_list[i])
            w_x = X_full[idx : idx + window_size] # (window_size, n_features)
            # Target aligns with the latest bucket of the window
            w_target = targets_full[idx + window_size - 1] # (14, num_classes)
            w_ts = timestamps_list[idx + window_size - 1]

            X_windows.append(w_x)
            targets_windows.append(w_target)
            ts_windows.append(w_ts)

        return np.array(X_windows, dtype=np.float32), torch.stack(targets_windows, dim=0), ts_windows

    # For window_size = 1, slices map directly 1-to-1
    # Trace/Metric/Log per split
    idx_train = list(range(0, n_train_windows))
    idx_val = list(range(n_train_windows, n_train_windows + n_val_windows))
    idx_test = list(range(n_train_windows + n_val_windows, num_total_buckets))

    def slice_split(indices: List[int], split_name: str) -> HolisticRCADataset:
        sub_trace = preprocessed_out.X_trace[indices]
        sub_metric = preprocessed_out.X_metric[indices]
        sub_log = preprocessed_out.X_log[indices]
        sub_targets = full_targets[indices]
        sub_ts = [unique_timestamps[i] for i in indices]

        if window_size == 1:
            w_trace = sub_trace[:, np.newaxis, :]  # (N, 1, 84)
            w_metric = sub_metric[:, np.newaxis, :] # (N, 1, 98)
            w_log = sub_log[:, np.newaxis, :]       # (N, 1, 84)
            w_targets = sub_targets                 # (N, 14, 3)
            w_ts = sub_ts
        else:
            w_trace, w_targets, w_ts = create_window_arrays(sub_trace, sub_targets, sub_ts)
            w_metric, _, _ = create_window_arrays(sub_metric, sub_targets, sub_ts)
            w_log, _, _ = create_window_arrays(sub_log, sub_targets, sub_ts)

        return HolisticRCADataset(
            X_trace=w_trace,
            X_metric=w_metric,
            X_log=w_log,
            targets=w_targets,
            edge_index=graph.edge_index,
            entity_indices=entity_indices,
            timestamps=w_ts,
            entities=entities,
            fault_classes=classes,
            split_name=split_name
        )

    train_dataset = slice_split(idx_train, "train")
    val_dataset = slice_split(idx_val, "val")
    test_dataset = slice_split(idx_test, "test")

    # 8. Compute Diagnostic Metadata per Split
    def build_split_meta(name: str, split_df: pd.DataFrame, ts_list: List[int]) -> SplitMetadata:
        pos_counts = {}
        neg_counts = {}
        unsupported = []
        total_rows = len(split_df)

        for fc in classes:
            pos = int(((split_df['is_fault'] == 1) & (split_df['fault_type'] == fc)).sum())
            neg = total_rows - pos
            pos_counts[fc] = pos
            neg_counts[fc] = neg
            if pos == 0:
                unsupported.append(fc)

        has_pos = any(v > 0 for v in pos_counts.values())
        return SplitMetadata(
            name=name,
            start_timestamp=ts_list[0],
            end_timestamp=ts_list[-1] + time_granularity,
            num_windows=len(ts_list),
            num_total_label_rows=total_rows,
            class_positive_counts=pos_counts,
            class_negative_counts=neg_counts,
            unsupported_classes=unsupported,
            has_positive_examples=has_pos
        )

    train_meta = build_split_meta("train", train_labels_df, train_ts)
    val_meta = build_split_meta("val", val_labels_df, val_ts)
    test_meta = build_split_meta("test", test_labels_df, test_ts)

    metadata = {
        'entities': entities,
        'fault_classes': classes,
        'train_log_vocab_size': len(train_log_vocab),
        'train_pos_weights': train_pos_weights,
        'feature_counts': preprocessed_out.feature_counts_by_type,
        'total_features': preprocessed_out.X_trace.shape[1] + preprocessed_out.X_metric.shape[1] + preprocessed_out.X_log.shape[1],
        'graph_num_nodes': len(entities),
        'graph_num_edges': graph.edge_index.shape[1],
        'splits': {
            'train': train_meta,
            'val': val_meta,
            'test': test_meta
        },
        'preprocessed_out': preprocessed_out
    }

    return train_dataset, val_dataset, test_dataset, metadata
