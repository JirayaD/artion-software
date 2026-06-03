"""
Spatiotemporal GraphSAGE Neural Engine for delay prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpatiotemporalGraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.2):
        """
        Spatiotemporal GraphSAGE for predicting delay propagation
        
        Args:
            input_dim: Number of input features per node
            hidden_dim: Hidden layer dimension
            output_dim: Number of output predictions (delay steps)
            num_layers: Number of GraphSAGE layers
            dropout: Dropout rate
        """
        super(SpatiotemporalGraphSAGE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Output layers for delay prediction
        self.linear1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear2 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        
        logger.info(f"Initialized SpatiotemporalGraphSAGE: {input_dim}->{hidden_dim}->{output_dim}")
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the GraphSAGE network
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector for pooling (optional)
        
        Returns:
            node_embeddings: [num_nodes, hidden_dim]
            graph_embedding: [batch_size, hidden_dim] (if batch provided)
            delay_predictions: [num_nodes, output_dim] or [batch_size, output_dim]
        """
        # GraphSAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # No activation on last layer
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        node_embeddings = x
        
        # Global pooling for graph-level representation (if needed)
        if batch is not None:
            graph_embedding = global_mean_pool(x, batch)
        else:
            graph_embedding = torch.mean(x, dim=0, keepdim=True)
        
        # Delay prediction head
        x = F.relu(self.linear1(node_embeddings))
        x = self.dropout_layer(x)
        delay_predictions = self.linear2(x)
        
        return node_embeddings, graph_embedding, delay_predictions
    
    def predict_delay(self, graph_data, node_features):
        """
        Predict delay propagation for the railway network
        
        Args:
            graph_data: NetworkX graph or PyTorch Geometric data
            node_features: Node feature matrix [num_nodes, input_dim]
        
        Returns:
            delay_predictions: [num_nodes, output_dim] delay predictions
        """
        self.eval()
        with torch.no_grad():
            # Convert NetworkX to edge_index if needed
            if hasattr(graph_data, 'edges'):
                import networkx as nx
                # Create mapping from node to index for proper edge_index format
                # Handle case where nodes are not integers (e.g., tuples)
                node_to_idx = {node: idx for idx, node in enumerate(graph_data.nodes())}
                edge_list = []
                for u, v in graph_data.edges():
                    edge_list.append([node_to_idx[u], node_to_idx[v]])
                if edge_list:
                    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = graph_data.edge_index if hasattr(graph_data, 'edge_index') else graph_data[1]
            
            # Ensure proper tensor types
            if not isinstance(node_features, torch.Tensor):
                node_features = torch.tensor(node_features, dtype=torch.float)
            
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, dtype=torch.long)
            
            # Forward pass
            _, _, delay_preds = self.forward(node_features, edge_index)
            
            return delay_preds.numpy()

# Utility functions for graph processing
def create_railway_graph_from_schedule(schedule_data):
    """
    Create a railway graph from schedule data
    
    Args:
        schedule_data: DataFrame or dict containing train schedules
    
    Returns:
        networkx.Graph: Railway network topology
    """
    import networkx as nx
    import pandas as pd
    
    G = nx.Graph()
    
    # This is a simplified version - in practice, you'd parse actual route data
    # For demonstration, creating a grid-like railway network
    # In real implementation, this would come from geographic/track data
    
    # Add nodes (stations/junctions)
    # Add edges (track connections)
    # Add edge attributes (distance, capacity, etc.)
    
    return G

def extract_spatiotemporal_features(schedule_data, current_time, horizon=5):
    """
    Extract spatiotemporal features for GraphSAGE input
    
    Args:
        schedule_data: Current schedule and train positions
        current_time: Current timestamp
        horizon: Prediction horizon
    
    Returns:
        node_features: Feature matrix for each node
    """
    # Features could include:
    # - Time of day (cyclical encoding)
    # - Day of week
    # - Historical delay patterns
    # - Current train density
    # - Weather conditions
    # - Special events
    
    # Placeholder implementation
    num_nodes = 20  # Example: 20 stations/junctions
    features = np.random.randn(num_nodes, 10)  # 10 features per node
    
    return features
