import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
from models.holistic_rca import HolisticRCA

def test_model_forward():
    batch_size = 2
    num_time_buckets = 5
    
    # Example dimensionalities
    feature_counts = {
        'service': 6, # from trace: 3 sets * 2 features
        'pod': 4,     # example metric count + some logs
        'node': 3     # example metric count
    }
    
    # 10 traces, 8 metrics, 15 log features total across the system
    total_trace_features = 10
    total_metric_features = 8
    total_log_features = 15
    
    # Synthetic batch of traces, metrics, logs
    traces = torch.rand(batch_size, num_time_buckets, total_trace_features)
    metrics = torch.rand(batch_size, num_time_buckets, total_metric_features)
    logs = torch.rand(batch_size, num_time_buckets, total_log_features)
    
    model = HolisticRCA(
        node_types=['service', 'pod', 'node'],
        feature_counts=feature_counts,
        num_time_buckets=num_time_buckets,
        embed_dim=32,
        entity_out_dim=64,
        num_fault_classes=5
    )
    
    # In a real scenario, entity_mappings maps the modality embeddings from H (embed_dim) 
    # to specific entities. Here we mock what HI_et would look like after gathering.
    # Suppose we have 2 services, 4 pods, and 1 node in the graph for each graph in batch.
    # Total nodes = (2 + 4 + 1) * batch_size = 14
    entity_mappings = {
        'service': torch.rand(batch_size * 2, feature_counts['service'], 32),
        'pod': torch.rand(batch_size * 4, feature_counts['pod'], 32),
        'node': torch.rand(batch_size * 1, feature_counts['node'], 32),
    }
    
    # RERG edges. Total nodes = 14. We create some random valid edges.
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 12, 13], 
        [2, 3, 4, 12, 13, 0, 1]
    ], dtype=torch.long)
    
    predictions = model(traces, metrics, logs, entity_mappings, edge_index)
    
    assert 'service' in predictions
    assert predictions['service'].shape == (batch_size * 2, 5) # 5 fault classes
    assert predictions['pod'].shape == (batch_size * 4, 5)
    assert predictions['node'].shape == (batch_size * 1, 5)
    print("Model forward pass completed successfully with correct output shapes!")

if __name__ == '__main__':
    test_model_forward()
