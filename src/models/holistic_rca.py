import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class ObservabilityRepresentation(nn.Module):
    """
    Section IV.B: Observability Representation Learning
    Eq (1): H = TE_mo(Linear_mo(X^T))
    """
    def __init__(self, num_time_buckets, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        # Maps time buckets (t) to embed_dim (\lambda)
        self.linear_map = nn.Linear(num_time_buckets, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: (batch_size, time_buckets, num_features)
        # Transpose to: (batch_size, num_features, time_buckets)
        x_t = x.transpose(1, 2)
        
        # Map time_buckets to embed_dim
        # x_mapped shape: (batch_size, num_features, embed_dim)
        x_mapped = self.linear_map(x_t)
        
        # Pass through Transformer
        # H shape: (batch_size, num_features, embed_dim)
        H = self.transformer_encoder(x_mapped)
        return H


class ResourceEntityFeatureIntegration(nn.Module):
    """
    Section IV.C: Resource Entity Feature Integration
    Eq (2): v_e = Flatten(Linear_et(TE_et(HI_e)))
    """
    def __init__(self, num_features, embed_dim=128, out_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # After TE, shape is (batch_size, num_features, embed_dim).
        # We flatten num_features and embed_dim, then map to out_dim (\nu).
        self.linear_out = nn.Linear(num_features * embed_dim, out_dim)

    def forward(self, HI_e):
        # HI_e shape: (batch_size, num_features, embed_dim)
        fused = self.transformer_encoder(HI_e)
        
        # Flatten the features and embedding dimensions
        fused_flat = torch.flatten(fused, start_dim=1)
        
        # Map to common dimension \nu
        v_e = self.linear_out(fused_flat)
        return v_e


class HolisticRCA(nn.Module):
    """
    HolisticRCA Model integrating Observability Representation,
    Resource Entity Integration, and Resource Entity Fusion (GAT).
    """
    def __init__(self, node_types, feature_counts, num_time_buckets, 
                 embed_dim=128, entity_out_dim=256, num_fault_classes=10):
        super().__init__()
        self.node_types = node_types
        
        # Modality specific representation learners (TE_mo)
        self.trace_rep = ObservabilityRepresentation(num_time_buckets, embed_dim)
        self.metric_rep = ObservabilityRepresentation(num_time_buckets, embed_dim)
        self.log_rep = ObservabilityRepresentation(num_time_buckets, embed_dim)
        
        # Entity type specific integrators (TE_et)
        self.entity_integrators = nn.ModuleDict({
            node_type: ResourceEntityFeatureIntegration(
                num_features=feature_counts[node_type], 
                embed_dim=embed_dim, 
                out_dim=entity_out_dim
            ) for node_type in node_types
        })
        
        # Resource Entity Fusion using Graph Attention Network
        self.gat = GATv2Conv(entity_out_dim, entity_out_dim, heads=4, concat=False)
        
        # Fault Classifiers for each entity type
        self.classifiers = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.ELU(),
                nn.Linear(entity_out_dim, num_fault_classes),
                # BCEWithLogitsLoss will be used, so no Sigmoid here to avoid numerical instability
            ) for node_type in node_types
        })

    def forward(self, traces, metrics, logs, entity_mappings, edge_index):
        """
        Forward pass for HolisticRCA.
        
        Args:
            traces, metrics, logs: Tensors of shape (batch_size, time_buckets, total_features_for_modality)
            entity_mappings: dict of node_type -> list of indices indicating which features 
                             belong to which entity. This maps the flat modality features 
                             into entity-specific feature groups (HI_e).
            edge_index: Tensor of shape (2, num_edges) for RERG.
        """
        # 1. Observability Representation Learning
        # H shapes: (batch_size, total_features, embed_dim)
        H_trace = self.trace_rep(traces) if traces is not None else None
        H_metric = self.metric_rep(metrics) if metrics is not None else None
        H_log = self.log_rep(logs) if logs is not None else None
        
        # 2. Resource Entity Feature Integration
        entity_embeddings = []
        entity_types = []
        
        # For simplicity in this batched structure, we assume we process all entities 
        # of a type together. In practice, we gather features from H for each entity.
        for node_type in self.node_types:
            # entity_mappings[node_type] contains the gathered feature tensors for this type
            # Shape expected: (batch_size * num_entities_of_type, num_features_of_type, embed_dim)
            HI_et = entity_mappings[node_type] 
            
            if HI_et is not None:
                # v_et shape: (batch_size * num_entities, entity_out_dim)
                v_et = self.entity_integrators[node_type](HI_et)
                entity_embeddings.append(v_et)
                entity_types.extend([node_type] * v_et.size(0))
                
        if not entity_embeddings:
            return {}
            
        # Combine all entity embeddings for the graph
        # v shape: (total_nodes, entity_out_dim)
        v = torch.cat(entity_embeddings, dim=0)
        
        # 3. Resource Entity Fusion
        v_prime = self.gat(v, edge_index)
        
        # 4. Fault Classification
        predictions = {}
        start_idx = 0
        for node_type in self.node_types:
            count = sum(1 for t in entity_types if t == node_type)
            if count > 0:
                end_idx = start_idx + count
                v_et_prime = v_prime[start_idx:end_idx]
                predictions[node_type] = self.classifiers[node_type](v_et_prime)
                start_idx = end_idx
                
        return predictions

