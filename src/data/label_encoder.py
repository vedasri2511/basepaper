"""
Label Encoder and Target Tensor Alignment for HolisticRCA.
Aligned with paper: "Holistic Root Cause Analysis for Failures in Cloud-Native Systems Through Observability Data"
(Yongqi Han et al., IEEE TSC 2024).

This module formalizes:
1. Phase D: Ground-truth label alignment with RERG node ordering and time windows.
2. Target binary tensor construction for multi-label fault classification (Section IV.E, Eq. 5).
3. Class weight calculation for paper-aligned Weighted BCE Loss (Section IV.E, Eq. 6).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

# Paper Section V.B.1: Dataset C (Social Network) defines 3 typical fault types for microservices:
# 1. CPU Load ('cpu_load')
# 2. Delay ('network_delay')
# 3. Packet Loss ('network_loss')
DEFAULT_SERVICE_FAULT_CLASSES = ['cpu_load', 'network_delay', 'network_loss']


class LabelEncoder:
    """
    Encodes tabular labels (timestamp, timestamp_end, entity, is_fault, fault_type)
    into aligned PyTorch target tensors matching the canonical node ordering of RERG.
    """
    def __init__(
        self,
        entities: Optional[List[str]] = None,
        fault_classes: Optional[List[str]] = None
    ):
        self.entities = list(entities) if entities is not None else None
        self.fault_classes = list(fault_classes) if fault_classes is not None else DEFAULT_SERVICE_FAULT_CLASSES

        self.fault_to_idx: Dict[str, int] = {fc: i for i, fc in enumerate(self.fault_classes)}
        self.idx_to_fault: Dict[int, str] = {i: fc for i, fc in enumerate(self.fault_classes)}

        if self.entities is not None:
            self.entity_to_idx: Dict[str, int] = {e: i for i, e in enumerate(self.entities)}
        else:
            self.entity_to_idx = {}

    def encode_label(self, is_fault: int, fault_type: str) -> np.ndarray:
        """
        Converts a single (is_fault, fault_type) record into a multi-label binary vector.

        Semantics:
        - is_fault == 0 or fault_type == 'normal' -> all zeros [0, ..., 0]
        - is_fault == 1 and fault_type in fault_classes -> one-hot vector for fault_type
        - is_fault == 1 and unknown fault_type -> raises ValueError
        """
        target = np.zeros(len(self.fault_classes), dtype=np.float32)
        if int(is_fault) == 0 or str(fault_type).lower() == 'normal':
            return target

        fault_type_str = str(fault_type).strip()
        if fault_type_str not in self.fault_to_idx:
            raise ValueError(
                f"Unknown fault type '{fault_type_str}'. Expected one of {self.fault_classes} or 'normal'."
            )

        target[self.fault_to_idx[fault_type_str]] = 1.0
        return target

    def create_window_target_tensor(
        self,
        labels_df: pd.DataFrame,
        time_start: int,
        time_end: Optional[int] = None,
        entities: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Creates an aligned target tensor of shape (|E|, num_classes) for a single time window.

        Requirements:
        - Preserves exact canonical entity ordering from `entities`.
        - Validates that every entity in `entities` is present in the window.
        - Rejects duplicate labels or ambiguous entries for the same entity in the same window.
        - Rejects unknown entities or unknown fault types.
        """
        target_entities = entities if entities is not None else self.entities
        if target_entities is None:
            raise ValueError("Entities list must be provided to ensure deterministic node ordering.")

        # Filter for window
        if time_end is not None:
            window_df = labels_df[
                (labels_df['timestamp'] == time_start) & (labels_df['timestamp_end'] == time_end)
            ]
        else:
            window_df = labels_df[labels_df['timestamp'] == time_start]

        if window_df.empty:
            raise ValueError(f"No labels found in labels_df for timestamp={time_start}, timestamp_end={time_end}")

        # Check for duplicate entity entries in this window
        entity_counts = window_df['entity'].value_counts()
        duplicates = entity_counts[entity_counts > 1].index.tolist()
        if duplicates:
            raise ValueError(f"Duplicate label entries found in time window for entities: {duplicates}")

        # Build lookup table: entity -> (is_fault, fault_type)
        row_lookup = {
            row['entity']: (row['is_fault'], row['fault_type'])
            for _, row in window_df.iterrows()
        }

        # Check for missing entities
        missing_entities = [e for e in target_entities if e not in row_lookup]
        if missing_entities:
            raise ValueError(
                f"Time window starting at {time_start} is missing labels for {len(missing_entities)} entities: {missing_entities[:5]}"
            )

        # Align targets in canonical entity order
        targets = []
        for entity in target_entities:
            is_fault, fault_type = row_lookup[entity]
            encoded = self.encode_label(is_fault, fault_type)
            targets.append(encoded)

        target_array = np.stack(targets, axis=0) # (|E|, num_classes)
        return torch.from_numpy(target_array).float()

    def create_sequence_target_tensor(
        self,
        labels_df: pd.DataFrame,
        entities: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Creates a sequence of aligned target tensors across all time windows in labels_df.

        Returns:
            targets: Tensor of shape (num_windows, |E|, num_classes)
            timestamps: List of window start timestamps
        """
        target_entities = entities if entities is not None else self.entities
        if target_entities is None:
            raise ValueError("Entities list must be provided.")

        unique_timestamps = sorted(labels_df['timestamp'].unique())
        sequence_targets = []

        for ts in unique_timestamps:
            window_target = self.create_window_target_tensor(
                labels_df=labels_df,
                time_start=ts,
                entities=target_entities
            )
            sequence_targets.append(window_target)

        return torch.stack(sequence_targets, dim=0), unique_timestamps

    def compute_class_weights(
        self,
        labels_df: pd.DataFrame,
        fault_classes: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Computes paper Eq. (6) class weights:
            p_c = N_{non-faulty, c} / N_{faulty, c}

        Safe edge-case handling:
        - If N_{faulty, c} == 0 (no positive samples for class c in dataset slice):
          p_c is safely set to 1.0 (avoids division by zero; positive loss term is 0 since y=0).
        - If N_{non-faulty, c} == 0 (all samples are faulty):
          p_c is safely set to 1.0.
        """
        classes = fault_classes if fault_classes is not None else self.fault_classes
        total_samples = len(labels_df)
        if total_samples == 0:
            return torch.ones(len(classes), dtype=torch.float32)

        weights = []
        for fc in classes:
            # Count positive instances for this fault type
            n_faulty = int(((labels_df['is_fault'] == 1) & (labels_df['fault_type'] == fc)).sum())
            n_non_faulty = total_samples - n_faulty

            if n_faulty == 0 or n_non_faulty == 0:
                p_c = 1.0
            else:
                p_c = float(n_non_faulty) / float(n_faulty)
            weights.append(p_c)

        return torch.tensor(weights, dtype=torch.float32)
