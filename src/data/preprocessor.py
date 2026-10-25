import os
import re
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


def natural_sort_key(s: str) -> List[Any]:
    """Helper for natural alphanumeric sorting (e.g. service_2 before service_10)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


@dataclass
class PreprocessedOutput:
    """
    Standardized container for preprocessed observability data.
    Aligned with Sections IV.A, IV.B, and IV.C of HolisticRCA.
    """
    X_trace: np.ndarray             # Shape: (num_buckets, n_trace_features)
    X_metric: np.ndarray            # Shape: (num_buckets, n_metric_features)
    X_log: np.ndarray               # Shape: (num_buckets, n_log_features)
    timestamps: List[int]           # Bucket start timestamps
    entities: List[str]             # Canonical ordered entity identifiers
    metric_types: List[str]         # Canonical ordered metric type names
    log_vocab: List[str]            # Extracted log vocabulary (templates and words)
    trace_feature_names: List[str]  # Detailed column names for X_trace
    metric_feature_names: List[str] # Detailed column names for X_metric
    log_feature_names: List[str]    # Detailed column names for X_log
    entity_feature_map: Dict[str, Dict[str, Any]] # Mapping per entity for tensor gathering
    feature_counts_by_type: Dict[str, int]        # Entity type -> number of features n_e


class Preprocessor:
    """
    Section IV.A: Observability Feature Engineering
    Handles trace, metric, and log feature engineering conforming to HolisticRCA.
    """
    def __init__(
        self,
        time_interval: Tuple[int, int],
        time_granularity: int,
        top_k_log: int = 5,
        tf_idf_threshold: float = 0.3,
        word_threshold: float = 0.04
    ):
        self.t_start, self.t_end = time_interval
        self.dt = time_granularity
        self.top_k_log = top_k_log
        self.tf_idf_threshold = tf_idf_threshold
        self.word_threshold = word_threshold
        config = TemplateMinerConfig()
        if os.path.exists("drain3.ini"):
            config.load("drain3.ini")
        self.template_miner = TemplateMiner(config=config)

    def partition_time(self) -> List[Tuple[int, int]]:
        """Partitions analysis window [t_start, t_end) into buckets of width dt."""
        num_buckets = int(math.ceil((self.t_end - self.t_start) / self.dt))
        return [(self.t_start + i * self.dt, self.t_start + (i + 1) * self.dt) for i in range(num_buckets)]

    def get_bucket_timestamps(self) -> List[int]:
        """Returns the start epoch timestamp for each bucket."""
        num_buckets = int(math.ceil((self.t_end - self.t_start) / self.dt))
        return [self.t_start + i * self.dt for i in range(num_buckets)]

    def _get_bucket_index(self, timestamp: float) -> int:
        """Maps an epoch timestamp to its 0-indexed bucket."""
        if timestamp < self.t_start or timestamp >= self.t_end:
            return -1
        return int((timestamp - self.t_start) // self.dt)

    def extract_trace_features(self, spans: pd.DataFrame, services: List[str]) -> np.ndarray:
        """
        Extracts 6 trace features per service per bucket (Section IV.A & Fig. 4):
        - S_curr: Average duration and cardinality (count) of spans executed in service s.
        - S_prev: Average duration and cardinality (count) of parent spans of S_curr.
        - S_next: Average duration and cardinality (count) of child spans of S_curr.

        spans: DataFrame with [span_id, parent_span_id, service_name, start_time, duration]
        services: list of unique service names
        Returns: numpy array of shape (num_buckets, len(services) * 6)
        """
        services = sorted(services, key=natural_sort_key)
        num_buckets = len(self.partition_time())
        features = np.zeros((num_buckets, len(services), 6), dtype=np.float32)

        if spans is None or spans.empty:
            return features.reshape((num_buckets, -1))

        # Build span lookup and child span index
        span_dict = {}
        child_dict = defaultdict(list)
        for _, row in spans.iterrows():
            s_id = str(row['span_id'])
            span_dict[s_id] = row
            p_id = row['parent_span_id']
            if pd.notna(p_id) and str(p_id) != '':
                child_dict[str(p_id)].append(row)

        service_to_idx = {srv: i for i, srv in enumerate(services)}

        # Group spans by (bucket_index, service_name)
        curr_spans = defaultdict(lambda: defaultdict(list))
        for _, row in spans.iterrows():
            b_idx = self._get_bucket_index(row['start_time'])
            srv = row['service_name']
            if b_idx != -1 and srv in service_to_idx:
                curr_spans[b_idx][srv].append(row)

        # Compute duration means and cardinality
        for b_idx in range(num_buckets):
            for srv_name, srv_idx in service_to_idx.items():
                s_curr = curr_spans[b_idx][srv_name]
                if not s_curr:
                    continue

                # 1. S_curr features (avg duration, count)
                features[b_idx, srv_idx, 0] = float(np.mean([s['duration'] for s in s_curr]))
                features[b_idx, srv_idx, 1] = float(len(s_curr))

                # 2. S_prev features (avg duration, count)
                s_prev = []
                for s in s_curr:
                    p_id = s['parent_span_id']
                    if pd.notna(p_id) and str(p_id) in span_dict:
                        s_prev.append(span_dict[str(p_id)])
                if s_prev:
                    features[b_idx, srv_idx, 2] = float(np.mean([s['duration'] for s in s_prev]))
                    features[b_idx, srv_idx, 3] = float(len(s_prev))

                # 3. S_next features (avg duration, count)
                s_next = []
                for s in s_curr:
                    c_id = str(s['span_id'])
                    if c_id in child_dict:
                        s_next.extend(child_dict[c_id])
                if s_next:
                    features[b_idx, srv_idx, 4] = float(np.mean([s['duration'] for s in s_next]))
                    features[b_idx, srv_idx, 5] = float(len(s_next))

        # Replace any potential NaNs or infs
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features.reshape((num_buckets, -1))

    def extract_metric_features(
        self,
        metrics_df: pd.DataFrame,
        entities: List[str],
        metric_types: List[str],
        flatten: bool = False
    ) -> np.ndarray:
        """
        Extracts metric features by calculating the mean for each bucket (Section IV.A).
        metrics_df: DataFrame with [timestamp, entity, metric_type, value]
        entities: ordered list of entity names
        metric_types: ordered list of metric type names
        flatten: If True, returns 2D matrix (num_buckets, len(entities) * len(metric_types)).
                 If False, returns 3D matrix (num_buckets, len(entities), len(metric_types)).
        """
        entities = sorted(entities, key=natural_sort_key)
        metric_types = sorted(metric_types)
        num_buckets = len(self.partition_time())
        features = np.zeros((num_buckets, len(entities), len(metric_types)), dtype=np.float32)

        if metrics_df is None or metrics_df.empty:
            return features.reshape((num_buckets, -1)) if flatten else features

        entity_to_idx = {e: i for i, e in enumerate(entities)}
        metric_to_idx = {m: i for i, m in enumerate(metric_types)}

        metrics_copy = metrics_df.copy()
        metrics_copy['b_idx'] = metrics_copy['timestamp'].apply(self._get_bucket_index)
        valid = metrics_copy[metrics_copy['b_idx'] != -1]

        if not valid.empty:
            grouped = valid.groupby(['b_idx', 'entity', 'metric_type'])['value'].mean().reset_index()
            for _, row in grouped.iterrows():
                b = int(row['b_idx'])
                e = row['entity']
                m = row['metric_type']
                if e in entity_to_idx and m in metric_to_idx:
                    features[b, entity_to_idx[e], metric_to_idx[m]] = float(row['value'])

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features.reshape((num_buckets, -1)) if flatten else features

    def build_log_vocabulary(self, logs_df: pd.DataFrame) -> List[str]:
        """
        Mines log events using Drain3 and extracts keywords via TF-IDF (Section IV.A).
        Returns a sorted, deterministic vocabulary of top log patterns and keywords.
        """
        if logs_df is None or logs_df.empty:
            return ['tmpl:EMPTY_LOG', 'word:none']

        num_buckets = len(self.partition_time())
        bucket_tmpls = defaultdict(list)
        bucket_words = defaultdict(list)

        for _, row in logs_df.iterrows():
            b_idx = self._get_bucket_index(row['timestamp'])
            if b_idx == -1:
                continue
            msg = str(row['message'])
            cluster = self.template_miner.add_log_message(msg)
            template = cluster['template_mined']
            bucket_tmpls[b_idx].append(template)

            # Extract alphanumeric words of length >= 3
            words = [w.lower().strip(': ,.!?') for w in msg.split()]
            words = [w for w in words if len(w) >= 3 and w.isalpha()]
            bucket_words[b_idx].extend(words)

        all_tmpls = sorted(list(set(t for tmpls in bucket_tmpls.values() for t in tmpls)))
        all_words = sorted(list(set(w for words in bucket_words.values() for w in words)))

        if not all_tmpls and not all_words:
            return ['tmpl:EMPTY_LOG', 'word:none']

        # Calculate TF-IDF per bucket
        N_b = max(1, num_buckets)
        tmpl_df = {t: sum(1 for b in range(N_b) if t in bucket_tmpls[b]) for t in all_tmpls}
        word_df = {w: sum(1 for b in range(N_b) if w in bucket_words[b]) for w in all_words}

        tmpl_scores = {}
        for t in all_tmpls:
            idf = math.log((1 + N_b) / (1 + tmpl_df[t])) + 1.0
            max_tf = max(
                (bucket_tmpls[b].count(t) / len(bucket_tmpls[b])) if bucket_tmpls[b] else 0.0
                for b in range(N_b)
            )
            tmpl_scores[t] = max_tf * idf

        word_scores = {}
        for w in all_words:
            idf = math.log((1 + N_b) / (1 + word_df[w])) + 1.0
            max_tf = max(
                (bucket_words[b].count(w) / len(bucket_words[b])) if bucket_words[b] else 0.0
                for b in range(N_b)
            )
            word_scores[w] = max_tf * idf

        # Filter by threshold and take top_k
        filtered_tmpls = [t for t, s in tmpl_scores.items() if s >= self.tf_idf_threshold]
        if not filtered_tmpls:
            filtered_tmpls = sorted(all_tmpls, key=lambda t: -tmpl_scores[t])[:self.top_k_log]
        else:
            filtered_tmpls = sorted(filtered_tmpls, key=lambda t: -tmpl_scores[t])[:self.top_k_log]

        filtered_words = [w for w, s in word_scores.items() if s >= self.word_threshold]
        if not filtered_words:
            filtered_words = sorted(all_words, key=lambda w: -word_scores[w])[:self.top_k_log]
        else:
            filtered_words = sorted(filtered_words, key=lambda w: -word_scores[w])[:self.top_k_log]

        vocab = [f"tmpl:{t}" for t in sorted(filtered_tmpls)] + [f"word:{w}" for w in sorted(filtered_words)]
        return sorted(vocab)

    def extract_log_features(
        self,
        logs_df: pd.DataFrame,
        entities: List[str],
        vocab: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extracts log occurrence features (Section IV.A).
        For each bucket and entity, counts occurrences of selected log templates and keywords.
        Returns:
            features: numpy array of shape (num_buckets, len(entities) * len(vocab))
            vocab: the vocabulary used
        """
        entities = sorted(entities, key=natural_sort_key)
        num_buckets = len(self.partition_time())

        if vocab is None:
            vocab = self.build_log_vocabulary(logs_df)

        vocab = sorted(vocab)
        features = np.zeros((num_buckets, len(entities), len(vocab)), dtype=np.float32)

        if logs_df is None or logs_df.empty:
            return features.reshape((num_buckets, -1)), vocab

        entity_to_idx = {e: i for i, e in enumerate(entities)}
        vocab_to_idx = {v: i for i, v in enumerate(vocab)}

        for _, row in logs_df.iterrows():
            b_idx = self._get_bucket_index(row['timestamp'])
            ent = row['entity']
            if b_idx == -1 or ent not in entity_to_idx:
                continue

            e_idx = entity_to_idx[ent]
            msg = str(row['message'])
            cluster = self.template_miner.add_log_message(msg)
            tmpl_key = f"tmpl:{cluster['template_mined']}"
            if tmpl_key in vocab_to_idx:
                features[b_idx, e_idx, vocab_to_idx[tmpl_key]] += 1.0

            words = [w.lower().strip(': ,.!?') for w in msg.split()]
            for w in words:
                word_key = f"word:{w}"
                if word_key in vocab_to_idx:
                    features[b_idx, e_idx, vocab_to_idx[word_key]] += 1.0

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features.reshape((num_buckets, -1)), vocab

    def process_all(
        self,
        raw_data: Dict[str, pd.DataFrame],
        entities: Optional[List[str]] = None,
        metric_types: Optional[List[str]] = None,
        log_vocab: Optional[List[str]] = None
    ) -> PreprocessedOutput:
        """
        Orchestrates feature extraction across all three modalities (Section IV.A).
        Produces X_trace, X_metric, X_log, timestamps, and entity_feature_map.
        """
        traces_df = raw_data.get('traces', pd.DataFrame())
        metrics_df = raw_data.get('metrics', pd.DataFrame())
        logs_df = raw_data.get('logs', pd.DataFrame())

        # Determine entities
        if entities is None:
            found = set()
            if not metrics_df.empty and 'entity' in metrics_df.columns:
                found.update(metrics_df['entity'].dropna().unique())
            if not traces_df.empty and 'service_name' in traces_df.columns:
                found.update(traces_df['service_name'].dropna().unique())
            if not logs_df.empty and 'entity' in logs_df.columns:
                found.update(logs_df['entity'].dropna().unique())
            entities = sorted(list(found), key=natural_sort_key)
        else:
            entities = sorted(entities, key=natural_sort_key)

        # Determine metric types
        if metric_types is None:
            if not metrics_df.empty and 'metric_type' in metrics_df.columns:
                metric_types = sorted(metrics_df['metric_type'].dropna().unique())
            else:
                metric_types = []
        else:
            metric_types = sorted(metric_types)

        # 1. Trace Features (6 per service)
        X_trace = self.extract_trace_features(traces_df, entities)
        trace_feature_names = [
            f"trace_{srv}_{f}"
            for srv in entities
            for f in ['curr_dur_avg', 'curr_count', 'prev_dur_avg', 'prev_count', 'next_dur_avg', 'next_count']
        ]

        # 2. Metric Features (len(metric_types) per entity)
        X_metric = self.extract_metric_features(metrics_df, entities, metric_types, flatten=True)
        metric_feature_names = [
            f"metric_{ent}_{m}"
            for ent in entities
            for m in metric_types
        ]

        # 3. Log Features (len(log_vocab) per entity)
        X_log, final_vocab = self.extract_log_features(logs_df, entities, vocab=log_vocab)
        log_feature_names = [
            f"log_{ent}_{v}"
            for ent in entities
            for v in final_vocab
        ]

        # Construct Entity Feature Map
        # For each entity, tracks indices in X_trace, X_metric, X_log, and global indices in concatenated H
        num_trace_feats = 6
        num_metric_feats = len(metric_types)
        num_log_feats = len(final_vocab)

        offset_metric = X_trace.shape[1]
        offset_log = offset_metric + X_metric.shape[1]

        entity_feature_map = {}
        for idx, ent in enumerate(entities):
            t_idx = list(range(idx * num_trace_feats, (idx + 1) * num_trace_feats))
            m_idx = list(range(idx * num_metric_feats, (idx + 1) * num_metric_feats))
            l_idx = list(range(idx * num_log_feats, (idx + 1) * num_log_feats))

            global_indices = (
                t_idx +
                [offset_metric + i for i in m_idx] +
                [offset_log + i for i in l_idx]
            )

            ent_features = (
                [trace_feature_names[i] for i in t_idx] +
                [metric_feature_names[i] for i in m_idx] +
                [log_feature_names[i] for i in l_idx]
            )

            entity_feature_map[ent] = {
                'entity_index': idx,
                'trace_indices': t_idx,
                'metric_indices': m_idx,
                'log_indices': l_idx,
                'global_indices': global_indices,
                'feature_names': ent_features,
                'total_features': len(global_indices)
            }

        timestamps = self.get_bucket_timestamps()
        features_per_service = num_trace_feats + num_metric_feats + num_log_feats
        feature_counts_by_type = {'service': features_per_service}

        return PreprocessedOutput(
            X_trace=X_trace,
            X_metric=X_metric,
            X_log=X_log,
            timestamps=timestamps,
            entities=entities,
            metric_types=metric_types,
            log_vocab=final_vocab,
            trace_feature_names=trace_feature_names,
            metric_feature_names=metric_feature_names,
            log_feature_names=log_feature_names,
            entity_feature_map=entity_feature_map,
            feature_counts_by_type=feature_counts_by_type
        )
