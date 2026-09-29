import torch
import torch.nn as nn
from torch.optim import Adam
from models.holistic_rca import HolisticRCA
from evaluate import evaluate_fault_classification

def train_holistic_rca(model, epochs=5, lr=0.001):
    """
    Synthetic Training loop for HolisticRCA.
    """
    optimizer = Adam(model.parameters(), lr=lr)
    
    # BCEWithLogitsLoss as the paper uses BCE for multi-label fault classification (Eq 6)
    criterion = nn.BCEWithLogitsLoss()
    
    # Synthetic dataset parameters
    batch_size = 2
    num_time_buckets = 5
    feature_counts = {'service': 6, 'pod': 4, 'node': 3}
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Generate Synthetic Data
        traces = torch.rand(batch_size, num_time_buckets, 10)
        metrics = torch.rand(batch_size, num_time_buckets, 8)
        logs = torch.rand(batch_size, num_time_buckets, 15)
        
        # Mappings matching our synthetic graph
        entity_mappings = {
            'service': torch.rand(batch_size * 2, feature_counts['service'], 32),
            'pod': torch.rand(batch_size * 4, feature_counts['pod'], 32),
            'node': torch.rand(batch_size * 1, feature_counts['node'], 32),
        }
        
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 12, 13], 
            [2, 3, 4, 12, 13, 0, 1]
        ], dtype=torch.long)
        
        # Synthetic Ground Truth Labels (Fault types)
        # Random binary labels for each entity (e.g. 5 classes)
        y_true = {
            'service': torch.empty(batch_size * 2, 5).random_(2),
            'pod': torch.empty(batch_size * 4, 5).random_(2),
            'node': torch.empty(batch_size * 1, 5).random_(2)
        }
        
        # 2. Forward Pass
        predictions_dict = model(traces, metrics, logs, entity_mappings, edge_index)
        
        # 3. Compute Loss (Eq 7)
        loss = 0.0
        num_classifiers = 0
        for node_type in model.node_types:
            if node_type in predictions_dict and predictions_dict[node_type].shape[0] > 0:
                preds = predictions_dict[node_type]
                true_labels = y_true[node_type]
                loss += criterion(preds, true_labels)
                num_classifiers += 1
                
        if num_classifiers > 0:
            loss = loss / num_classifiers
            
        # 4. Backward Pass
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        # Print Evaluation Metrics on the last epoch
        if epoch == epochs - 1:
            print("\n--- Final Epoch Evaluation (Dimension 3) ---")
            for node_type in model.node_types:
                if node_type in predictions_dict:
                    # Apply sigmoid for eval since we used BCEWithLogitsLoss
                    probs = torch.sigmoid(predictions_dict[node_type]).detach().numpy()
                    true_labels_np = y_true[node_type].numpy()
                    metrics_result = evaluate_fault_classification(true_labels_np, probs)
                    print(f"Entity: {node_type}, Metrics: {metrics_result}")

if __name__ == '__main__':
    node_types = ['service', 'pod', 'node']
    feature_counts = {'service': 6, 'pod': 4, 'node': 3}
    
    # Initialize the model
    model = HolisticRCA(
        node_types=node_types, 
        feature_counts=feature_counts,
        num_time_buckets=5,
        embed_dim=32,
        entity_out_dim=64,
        num_fault_classes=5
    )
    print("Starting Synthetic Training...")
    train_holistic_rca(model, epochs=5)
