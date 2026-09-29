import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskLearner(nn.Module):
    """
    Section IV.F: Mask Learning
    Enables holistic RCA by learning masks for resource entity relations and observability features.
    This explains the model's predictions by identifying the most influential features/edges.
    """
    def __init__(self, trained_model, num_nodes, num_features):
        super().__init__()
        self.model = trained_model
        
        # Freeze all parameters of the saved model
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Learnable mask for resource entity relations in RERG (Eq 8)
        # Shape: (num_nodes, num_nodes)
        self.EM = nn.Parameter(torch.randn(num_nodes, num_nodes))
        
        # Learnable mask for observability features (Eq 8)
        # Shape: (num_features,)
        self.om = nn.Parameter(torch.randn(num_features))
        
    def forward(self, traces, metrics, logs, entity_mappings, edge_index):
        # 1. Apply observability mask (X \odot om)
        # We apply the sigmoid of om to keep masks between 0 and 1
        feature_mask = torch.sigmoid(self.om)
        
        masked_traces = traces * feature_mask if traces is not None else None
        masked_metrics = metrics * feature_mask if metrics is not None else None
        masked_logs = logs * feature_mask if logs is not None else None
        
        # 2. Apply topology mask (G \odot \sigma(EM))
        # Edge mask applied to edge weights or attention weights in a real implementation.
        # For simplicity, we just extract the relevant mask values for the existing edges.
        row, col = edge_index
        edge_mask = torch.sigmoid(self.EM[row, col])
        
        # 3. Reconstruct model input and observe differences (predictions)
        # Note: GATv2Conv natively accepts edge_weight if modified, or we can multiply v_prime.
        # To strictly follow the paper, edge_mask weights the message passing. 
        # Here we pass it as a structural skeleton.
        predictions = self.model(
            masked_traces, 
            masked_metrics, 
            masked_logs, 
            entity_mappings, 
            edge_index
        )
        
        return predictions, edge_mask, feature_mask

    def loss_function(self, masked_preds, original_preds, edge_mask, feature_mask, size_reg=1e-4, entropy_reg=1e-4):
        """
        Calculates the Mutual Information loss based on Eq 8.
        The goal is to maximize MI (minimize -MI), keeping the predictions close while minimizing the mask size.
        """
        # Cross entropy between original predictions and masked predictions
        mi_loss = 0
        for node_type in masked_preds:
            # Maximizing mutual information approx: minimize difference
            mi_loss += F.mse_loss(masked_preds[node_type], original_preds[node_type])
            
        # Regularization to keep masks sparse (identifying only the most critical features/edges)
        m_size = edge_mask.sum() + feature_mask.sum()
        
        # Entropy to encourage mask values towards 0 or 1
        m_ent = -edge_mask * torch.log(edge_mask + 1e-15) - (1 - edge_mask) * torch.log(1 - edge_mask + 1e-15)
        f_ent = -feature_mask * torch.log(feature_mask + 1e-15) - (1 - feature_mask) * torch.log(1 - feature_mask + 1e-15)
        
        total_loss = mi_loss + size_reg * m_size + entropy_reg * (m_ent.mean() + f_ent.mean())
        return total_loss
