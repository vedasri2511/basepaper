import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
from models.holistic_rca import HolisticRCA
from models.mask_learning import MaskLearner

def test_mask_learning():
    batch_size = 1
    num_time_buckets = 5
    num_nodes = 7 # 2 service, 4 pod, 1 node
    
    # 10 traces, 8 metrics, 15 log features total = 33 total max
    total_features = 33
    
    feature_counts = {
        'service': 6,
        'pod': 4,
        'node': 3
    }
    
    # Create the model
    model = HolisticRCA(
        node_types=['service', 'pod', 'node'],
        feature_counts=feature_counts,
        num_time_buckets=num_time_buckets,
        embed_dim=32,
        entity_out_dim=64,
        num_fault_classes=5
    )
    
    # Create the Mask Learner
    mask_learner = MaskLearner(model, num_nodes=num_nodes, num_features=total_features)
    
    # Fake inputs
    traces = torch.rand(batch_size, num_time_buckets, total_features)
    metrics = torch.rand(batch_size, num_time_buckets, total_features)
    logs = torch.rand(batch_size, num_time_buckets, total_features)
    
    entity_mappings = {
        'service': torch.rand(batch_size * 2, feature_counts['service'], 32),
        'pod': torch.rand(batch_size * 4, feature_counts['pod'], 32),
        'node': torch.rand(batch_size * 1, feature_counts['node'], 32),
    }
    
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6], 
        [2, 3, 4, 5, 6, 0, 1]
    ], dtype=torch.long)
    
    # Get original predictions
    original_preds = model(traces, metrics, logs, entity_mappings, edge_index)
    
    # Run through mask learner
    masked_preds, edge_mask, feature_mask = mask_learner(traces, metrics, logs, entity_mappings, edge_index)
    
    # Calc loss
    loss = mask_learner.loss_function(masked_preds, original_preds, edge_mask, feature_mask)
    
    assert loss.item() >= 0
    assert edge_mask.shape[0] == edge_index.shape[1]
    print("Mask Learning forward and loss calculation successful!")

if __name__ == '__main__':
    test_mask_learning()
