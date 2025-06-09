"""
Graph Neural Network Models for ʻŌhanaGPT using MLX (Apple Silicon)
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List, NamedTuple
from dataclasses import dataclass


class GraphData(NamedTuple):
    """Graph data structure for MLX"""
    x: mx.array  # Node features
    edge_index: mx.array  # Edge indices
    edge_attr: Optional[mx.array] = None  # Edge attributes
    y: Optional[mx.array] = None  # Labels
    train_mask: Optional[mx.array] = None
    val_mask: Optional[mx.array] = None
    test_mask: Optional[mx.array] = None


class MessagePassing(nn.Module):
    """Base class for message passing in MLX"""
    
    def __init__(self, aggr: str = 'mean'):
        super().__init__()
        self.aggr = aggr
    
    def forward(self, x: mx.array, edge_index: mx.array, edge_attr: Optional[mx.array] = None) -> mx.array:
        # Get source and target node indices
        row, col = edge_index[0], edge_index[1]
        
        # Compute messages
        messages = self.message(x[row], x[col], edge_attr)
        
        # Aggregate messages
        out = self.aggregate(messages, col, x.shape[0])
        
        # Update node features
        return self.update(out, x)
    
    def message(self, x_i: mx.array, x_j: mx.array, edge_attr: Optional[mx.array]) -> mx.array:
        """Compute messages from neighbors"""
        return x_j  # Default: pass neighbor features
    
    def aggregate(self, messages: mx.array, index: mx.array, num_nodes: int) -> mx.array:
        """Aggregate messages"""
        # Create output array
        out = mx.zeros((num_nodes, messages.shape[-1]))
        
        # Aggregate based on method
        if self.aggr == 'mean':
            # Count messages per node
            ones = mx.ones((messages.shape[0], 1))
            count = mx.zeros((num_nodes, 1))
            
            # Sum messages and counts
            for i in range(messages.shape[0]):
                out[index[i]] += messages[i]
                count[index[i]] += ones[i]
            
            # Average
            out = out / mx.maximum(count, 1.0)
        elif self.aggr == 'sum':
            for i in range(messages.shape[0]):
                out[index[i]] += messages[i]
        elif self.aggr == 'max':
            out = mx.full((num_nodes, messages.shape[-1]), float('-inf'))
            for i in range(messages.shape[0]):
                out[index[i]] = mx.maximum(out[index[i]], messages[i])
        
        return out
    
    def update(self, aggr_out: mx.array, x: mx.array) -> mx.array:
        """Update node features"""
        return aggr_out


class GATConv(MessagePassing):
    """Graph Attention Network layer for MLX"""
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 8, 
                 concat: bool = True, dropout: float = 0.0):
        super().__init__(aggr='sum')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        
        # Linear transformations
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention parameters
        self.att_src = nn.Linear(out_channels, 1, bias=False)
        self.att_dst = nn.Linear(out_channels, 1, bias=False)
        
        # Bias
        if concat:
            self.bias = mx.zeros((heads * out_channels,))
        else:
            self.bias = mx.zeros((out_channels,))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Initialize parameters
        gain = mx.sqrt(2.0)
        self.lin.weight = self.lin.weight * gain
        self.att_src.weight = self.att_src.weight * gain
        self.att_dst.weight = self.att_dst.weight * gain
    
    def forward(self, x: mx.array, edge_index: mx.array, edge_attr: Optional[mx.array] = None) -> mx.array:
        # Linear transformation
        x = self.lin(x)
        x = x.reshape(x.shape[0], self.heads, self.out_channels)
        
        # Get source and target indices
        row, col = edge_index[0], edge_index[1]
        
        # Compute attention scores
        alpha_src = self.att_src(x).squeeze(-1)
        alpha_dst = self.att_dst(x).squeeze(-1)
        alpha = alpha_src[row] + alpha_dst[col]
        alpha = mx.nn.leaky_relu(alpha, negative_slope=0.2)
        
        # Softmax over neighbors
        alpha = self._softmax(alpha, col, x.shape[0])
        
        # Apply dropout
        if self.training and self.dropout > 0:
            alpha = mx.dropout(alpha, self.dropout)
        
        # Message passing
        messages = x[row] * alpha.reshape(-1, self.heads, 1)
        out = self.aggregate(messages, col, x.shape[0])
        
        # Concatenate or average heads
        if self.concat:
            out = out.reshape(out.shape[0], -1)
        else:
            out = out.mean(axis=1)
        
        # Add bias
        out = out + self.bias
        
        return out
    
    def _softmax(self, alpha: mx.array, index: mx.array, num_nodes: int) -> mx.array:
        """Compute softmax over neighborhoods"""
        # This is a simplified version - in practice, you'd need scatter operations
        alpha_exp = mx.exp(alpha - mx.max(alpha))
        
        # Sum exponentials per node
        alpha_sum = mx.zeros((num_nodes, alpha.shape[1]))
        for i in range(alpha.shape[0]):
            alpha_sum[index[i]] += alpha_exp[i]
        
        # Normalize
        alpha_norm = alpha_exp / mx.maximum(alpha_sum[index], 1e-16)
        
        return alpha_norm


class GraphSAGE(MessagePassing):
    """GraphSAGE layer for MLX"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 normalize: bool = True, bias: bool = True):
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        
        # Linear layers
        self.lin_self = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x: mx.array, edge_index: mx.array, edge_attr: Optional[mx.array] = None) -> mx.array:
        # Message passing for neighbor aggregation
        neigh_out = super().forward(x, edge_index, edge_attr)
        neigh_out = self.lin_neigh(neigh_out)
        
        # Self connection
        self_out = self.lin_self(x)
        
        # Combine
        out = self_out + neigh_out
        
        # Normalize
        if self.normalize:
            out = out / mx.maximum(mx.linalg.norm(out, axis=-1, keepdims=True), 1e-16)
        
        return out


class FamilyGNN_MLX(nn.Module):
    """Graph Neural Network for family relationship prediction in MLX"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initial transformation
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = []
        self.layer_norms = []
        
        for i in range(num_layers):
            if i == 0:
                layer = GATConv(hidden_dim, hidden_dim // num_heads, 
                               heads=num_heads, dropout=dropout, concat=True)
            elif i == num_layers - 1:
                layer = GATConv(hidden_dim, output_dim, 
                               heads=1, dropout=dropout, concat=False)
            else:
                layer = GATConv(hidden_dim, hidden_dim // num_heads,
                               heads=num_heads, dropout=dropout, concat=True)
            
            self.gnn_layers.append(layer)
            
            # Layer normalization
            if i < num_layers - 1:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
            else:
                self.layer_norms.append(nn.LayerNorm(output_dim))
        
        # Skip connection
        self.skip_proj = nn.Linear(hidden_dim, output_dim)
    
    def __call__(self, x: mx.array, edge_index: mx.array, 
                 edge_attr: Optional[mx.array] = None) -> mx.array:
        # Initial projection
        x = self.input_proj(x)
        x = mx.nn.relu(x)
        
        if self.training and self.dropout > 0:
            x = mx.dropout(x, self.dropout)
        
        # Store for skip connection
        x_init = x
        
        # Apply GNN layers
        for i, (gnn, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            x = gnn(x, edge_index, edge_attr)
            x = norm(x)
            
            if i < self.num_layers - 1:
                x = mx.nn.relu(x)
                if self.training and self.dropout > 0:
                    x = mx.dropout(x, self.dropout)
        
        # Add skip connection
        x = x + self.skip_proj(x_init)
        
        return x


class ParentPredictionGNN_MLX(nn.Module):
    """Complete model for predicting missing parents in MLX"""
    
    def __init__(self,
                 node_feature_dim: int,
                 hidden_dim: int = 256,
                 embedding_dim: int = 128,
                 num_gnn_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        
        # GNN backbone
        self.gnn = FamilyGNN_MLX(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_gnn_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Parent prediction heads
        self.father_predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mother_predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Compatibility scorer (simplified for MLX)
        self.compat_query = nn.Linear(hidden_dim, embedding_dim)
        self.compat_key = nn.Linear(embedding_dim, embedding_dim)
    
    def __call__(self, data: GraphData) -> Dict[str, mx.array]:
        # Get node embeddings
        node_embeddings = self.gnn(data.x, data.edge_index, data.edge_attr)
        
        # Get query nodes
        if self.training:
            query_mask = data.train_mask
        else:
            query_mask = data.test_mask
        
        query_indices = mx.where(query_mask)[0]
        query_embeddings = node_embeddings[query_indices]
        
        # Generate parent candidate representations
        father_queries = self.father_predictor(query_embeddings)
        mother_queries = self.mother_predictor(query_embeddings)
        
        # Score all possible parents
        father_scores = self._compute_parent_scores(
            father_queries, node_embeddings, data, 'father'
        )
        mother_scores = self._compute_parent_scores(
            mother_queries, node_embeddings, data, 'mother'
        )
        
        return {
            'father_scores': father_scores,
            'mother_scores': mother_scores,
            'node_embeddings': node_embeddings,
            'query_indices': query_indices
        }
    
    def _compute_parent_scores(self, 
                              queries: mx.array,
                              all_embeddings: mx.array,
                              data: GraphData,
                              parent_type: str) -> mx.array:
        """Compute scores for all possible parents"""
        # Transform queries and keys
        Q = self.compat_query(queries)
        K = self.compat_key(all_embeddings)
        
        # Compute scores via dot product
        scores = mx.matmul(Q, K.T) / mx.sqrt(Q.shape[-1])
        
        # Apply constraints
        scores = self._apply_constraints(scores, data, parent_type)
        
        return scores
    
    def _apply_constraints(self, 
                          scores: mx.array,
                          data: GraphData,
                          parent_type: str) -> mx.array:
        """Apply logical constraints to parent scores"""
        # Gender constraint
        if hasattr(data, 'gender') and data.gender is not None:
            if parent_type == 'father':
                # Mask out non-male candidates
                male_mask = (data.gender == 1)
                scores = mx.where(male_mask[None, :], scores, -float('inf'))
            else:  # mother
                # Mask out non-female candidates
                female_mask = (data.gender == 0)
                scores = mx.where(female_mask[None, :], scores, -float('inf'))
        
        # Generation constraint
        if hasattr(data, 'generation') and data.generation is not None:
            # Parents should be ~1 generation older
            query_gen = data.generation[data.query_indices]
            gen_diff = query_gen[:, None] - data.generation[None, :]
            invalid_gen = (gen_diff < 0.5) | (gen_diff > 2.0)
            scores = mx.where(~invalid_gen, scores, -float('inf'))
        
        return scores
    
    def predict_parents(self, data: GraphData, top_k: int = 5) -> Dict[str, mx.array]:
        """Predict top-k most likely parents for each individual"""
        outputs = self(data)
        
        # Get top-k candidates
        father_topk = mx.topk(outputs['father_scores'], k=top_k, axis=1)
        mother_topk = mx.topk(outputs['mother_scores'], k=top_k, axis=1)
        
        return {
            'father_indices': father_topk[1],  # indices
            'father_scores': father_topk[0],   # values
            'mother_indices': mother_topk[1],
            'mother_scores': mother_topk[0],
            'query_indices': outputs['query_indices']
        }


class RelationalGNN_MLX(nn.Module):
    """Relational GNN that handles different edge types"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 num_edge_types: int = 4,
                 num_layers: int = 3):
        super().__init__()
        
        self.num_edge_types = num_edge_types
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Edge type embeddings
        self.edge_embeddings = nn.Embedding(num_edge_types, hidden_dim)
        
        # Relational convolution layers
        self.conv_layers = []
        for i in range(num_layers):
            if i == num_layers - 1:
                self.conv_layers.append(
                    RelationalGraphConv(hidden_dim, output_dim, num_edge_types)
                )
            else:
                self.conv_layers.append(
                    RelationalGraphConv(hidden_dim, hidden_dim, num_edge_types)
                )
    
    def __call__(self, x: mx.array, edge_index: mx.array, edge_type: mx.array) -> mx.array:
        # Initial projection
        x = self.input_proj(x)
        x = mx.nn.relu(x)
        
        # Apply relational convolutions
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_type)
            x = mx.nn.relu(x)
        
        return x


class RelationalGraphConv(MessagePassing):
    """Relational Graph Convolution layer"""
    
    def __init__(self, in_channels: int, out_channels: int, num_relations: int):
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        
        # Relation-specific transformations
        self.weight = mx.zeros((num_relations, in_channels, out_channels))
        self.bias = mx.zeros((out_channels,))
        
        # Self-loop transformation
        self.self_loop = nn.Linear(in_channels, out_channels)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Initialize weights
        for r in range(self.num_relations):
            self.weight[r] = mx.random.normal(
                (self.in_channels, self.out_channels)
            ) * np.sqrt(2.0 / self.in_channels)
    
    def forward(self, x: mx.array, edge_index: mx.array, edge_type: mx.array) -> mx.array:
        # Self-loop
        out = self.self_loop(x)
        
        # Process each relation type
        for r in range(self.num_relations):
            # Get edges of this type
            mask = (edge_type == r)
            if mx.sum(mask) > 0:
                r_edge_index = edge_index[:, mask]
                
                # Transform features
                x_transformed = mx.matmul(x, self.weight[r])
                
                # Message passing for this relation
                r_out = super().forward(x_transformed, r_edge_index)
                out = out + r_out
        
        # Add bias
        out = out + self.bias
        
        return out


def create_mlx_graph_data(individuals_dict, families_dict, 
                         node_features: np.ndarray,
                         edge_index: np.ndarray,
                         edge_attr: Optional[np.ndarray] = None) -> GraphData:
    """Create MLX graph data from numpy arrays"""
    
    # Convert to MLX arrays
    x = mx.array(node_features)
    edge_index = mx.array(edge_index)
    
    if edge_attr is not None:
        edge_attr = mx.array(edge_attr)
    
    # Create masks
    num_nodes = len(individuals_dict)
    train_mask = mx.zeros(num_nodes, dtype=mx.bool_)
    val_mask = mx.zeros(num_nodes, dtype=mx.bool_)
    test_mask = mx.zeros(num_nodes, dtype=mx.bool_)
    
    # Simple split - in practice, use the deduplication info
    indices = mx.arange(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    return GraphData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )