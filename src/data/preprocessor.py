import pandas as pd
import numpy as np
from collections import defaultdict
import math
from drain3 import TemplateMiner # Assuming drain3 for Drain log parsing

class Preprocessor:
    """
    Section IV.A: Observability Feature Engineering
    Handles trace, metric, and log feature engineering.
    """
    def __init__(self, time_interval, time_granularity):
        self.t_start, self.t_end = time_interval
        self.dt = time_granularity
        self.template_miner = TemplateMiner()
        
    def partition_time(self):
        """Partitions time into multiple buckets based on dt."""
        num_buckets = int(np.ceil((self.t_end - self.t_start) / self.dt))
        buckets = [(self.t_start + i * self.dt, self.t_start + (i + 1) * self.dt) for i in range(num_buckets)]
        return buckets

    def _get_bucket_index(self, timestamp):
        if timestamp < self.t_start or timestamp >= self.t_end:
            return -1
        return int((timestamp - self.t_start) // self.dt)

    def extract_trace_features(self, spans, services):
        """
        Extracts trace features: S_curr, S_prev, S_next and workload intensity.
        Calculates average duration for each set in a bucket.
        spans: DataFrame with [span_id, parent_span_id, service_name, start_time, duration]
        services: list of unique service names
        Returns: numpy array of shape (num_buckets, len(services) * 6)
        """
        num_buckets = len(self.partition_time())
        features = np.zeros((num_buckets, len(services), 6))
        
        # Build index for fast lookup
        span_dict = {row['span_id']: row for _, row in spans.iterrows()}
        child_dict = defaultdict(list)
        for _, row in spans.iterrows():
            if pd.notna(row['parent_span_id']):
                child_dict[row['parent_span_id']].append(row)
                
        service_to_idx = {srv: i for i, srv in enumerate(services)}
        
        # Organize spans by bucket and service
        curr_spans_by_bucket_service = defaultdict(lambda: defaultdict(list))
        for _, row in spans.iterrows():
            b_idx = self._get_bucket_index(row['start_time'])
            if b_idx != -1 and row['service_name'] in service_to_idx:
                curr_spans_by_bucket_service[b_idx][row['service_name']].append(row)
                
        for b_idx in range(num_buckets):
            for srv_name, srv_idx in service_to_idx.items():
                s_curr = curr_spans_by_bucket_service[b_idx][srv_name]
                
                if not s_curr:
                    continue
                    
                # S_curr features
                features[b_idx, srv_idx, 0] = np.mean([s['duration'] for s in s_curr])
                features[b_idx, srv_idx, 1] = len(s_curr)
                
                # S_prev features
                s_prev = []
                for s in s_curr:
                    p_id = s['parent_span_id']
                    if pd.notna(p_id) and p_id in span_dict:
                        s_prev.append(span_dict[p_id])
                if s_prev:
                    features[b_idx, srv_idx, 2] = np.mean([s['duration'] for s in s_prev])
                    features[b_idx, srv_idx, 3] = len(s_prev)
                    
                # S_next features
                s_next = []
                for s in s_curr:
                    c_id = s['span_id']
                    if c_id in child_dict:
                        s_next.extend(child_dict[c_id])
                if s_next:
                    features[b_idx, srv_idx, 4] = np.mean([s['duration'] for s in s_next])
                    features[b_idx, srv_idx, 5] = len(s_next)
                    
        # Flatten features for each service: (num_buckets, num_services * 6)
        return features.reshape((num_buckets, -1))

    def extract_metric_features(self, metrics_df, entities, metric_types):
        """
        Extracts metric features by calculating the mean for each bucket.
        metrics_df: DataFrame with [timestamp, entity, metric_type, value]
        """
        num_buckets = len(self.partition_time())
        features = np.zeros((num_buckets, len(entities), len(metric_types)))
        
        entity_to_idx = {e: i for i, e in enumerate(entities)}
        metric_to_idx = {m: i for i, m in enumerate(metric_types)}
        
        # Add bucket index column
        metrics_df['b_idx'] = metrics_df['timestamp'].apply(self._get_bucket_index)
        valid_metrics = metrics_df[metrics_df['b_idx'] != -1]
        
        # Group and calculate mean
        grouped = valid_metrics.groupby(['b_idx', 'entity', 'metric_type'])['value'].mean().reset_index()
        
        for _, row in grouped.iterrows():
            if row['entity'] in entity_to_idx and row['metric_type'] in metric_to_idx:
                features[
                    int(row['b_idx']), 
                    entity_to_idx[row['entity']], 
                    metric_to_idx[row['metric_type']]
                ] = row['value']
                
        return features

    def extract_log_features(self, logs_df, entities, top_k=10, tf_idf_threshold=0.3):
        """
        Parses logs using Drain, computes TF-IDF for log events and keywords,
        and selects top_k features exceeding threshold.
        logs_df: DataFrame with [timestamp, entity, message]
        """
        num_buckets = len(self.partition_time())
        logs_df['b_idx'] = logs_df['timestamp'].apply(self._get_bucket_index)
        valid_logs = logs_df[logs_df['b_idx'] != -1].copy()
        
        # 1. Parse logs using Drain
        valid_logs['template'] = valid_logs['message'].apply(
            lambda msg: self.template_miner.add_log_message(msg)["template_mined"]
        )
        # 2. Extract words
        valid_logs['words'] = valid_logs['message'].str.split()
        
        # 3. TF-IDF calculation (simplified per bucket over all entities)
        # In a real implementation, we would construct a TF-IDF matrix per bucket
        # and filter by tf_idf_threshold.
        # For this skeleton, we represent the structure.
        
        # Placeholder for final log features shape: (num_buckets, len(entities), num_selected_features)
        # Where num_selected_features is at most top_k * 2 (events + words)
        pass

    def process_all(self, raw_data):
        """Orchestrates feature extraction for all modalities."""
        # traces = self.extract_trace_features(raw_data['traces'])
        # metrics = self.extract_metric_features(raw_data['metrics'])
        # logs = self.extract_log_features(raw_data['logs'])
        return {}
