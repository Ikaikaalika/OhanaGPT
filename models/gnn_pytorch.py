"""
Graph Neural Network Models for ʻŌhanaGPT using PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphSAGE, GCNConv, global_mean_pool, HeteroConv
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, HeteroData
from typing import Dict, Optional, Tuple, List
import math


class RelationalGATLayer(MessagePassing):
    """Custom GAT layer that handles relational information"""
    
    def __init__(self, in_channels: int, out_channels: int, num_relations: int = 3, heads: int = 8):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.num_relations = num_relations
        
        # Relation-specific transformations
        self.relation_lin = nn.ModuleList([
            nn.Linear(in_channels, out_channels * heads, bias=False)
            for _ in range(num_relations)
        ])
        
        # Attention parameters
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels * heads))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for lin in self.relation_lin:
            nn.init.xavier_uniform_(lin.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_attr=None):
        # Transform node features based on relation type
        if edge_attr is not None:
            # edge_attr contains relation type info
            rel_type = edge_attr.argmax(dim=1)  # Get relation type
            
            # Apply relation-specific transformations
            x_transformed = torch.zeros(x.size(0), self.out_channels * self.heads, device=x.device)
            for r in range(self.num_relations):
                mask = (rel_type == r)
                if mask.any():
                    x_transformed += self.relation_lin[r](x)
        else:
            x_transformed = self.relation_lin[0](x)
        
        return self.propagate(edge_index, x=x_transformed)
    
    def message(self, x_i, x_j, index, ptr, size_i):
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        
        # Compute attention coefficients
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = F.softmax(alpha, dim=0)
        
        return (x_j * alpha.unsqueeze(-1)).view(-1, self.heads * self.out_channels)
    
    def update(self, aggr_out):
        return aggr_out + self.bias


class FamilyGNN(nn.Module):
    """Graph Neural Network for family relationship prediction"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.2,
                 use_edge_attr: bool = True):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr
        
        # Initial transformation
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gnn_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, 
                           heads=num_heads, dropout=dropout, concat=True)
                )
            elif i == num_layers - 1:
                self.gnn_layers.append(
                    GATConv(hidden_dim, output_dim, 
                           heads=1, dropout=dropout, concat=False)
                )
            else:
                self.gnn_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads,
                           heads=num_heads, dropout=dropout, concat=True)
                )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim if i < num_layers - 1 else output_dim)
            for i in range(num_layers)
        ])
        
        # Skip connections
        self.skip_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Initial projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Store for skip connection
        x_init = x
        
        # Apply GNN layers
        for i, (gnn, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            x = gnn(x, edge_index)
            x = norm(x)
            
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Add skip connection
        if x.shape[1] != x_init.shape[1]:
            x = x + self.skip_proj(x_init)
        
        return x


class ParentPredictionGNN(nn.Module):
    """Complete model for predicting missing parents"""
    
    def __init__(self,
                 node_feature_dim: int,
                 hidden_dim: int = 256,
                 embedding_dim: int = 128,
                 num_gnn_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        
        # GNN backbone
        self.gnn = FamilyGNN(
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
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.mother_predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Compatibility scorer
        self.compatibility_scorer = nn.Bilinear(hidden_dim, embedding_dim, 1)
        
        # Generation constraint
        self.generation_embedder = nn.Embedding(10, 16)  # Max 10 generation differences
        self.generation_scorer = nn.Linear(16, 1)
    
    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        # Get node embeddings
        node_embeddings = self.gnn(data.x, data.edge_index, data.edge_attr)
        
        # Get query nodes (nodes needing parent prediction)
        query_mask = data.test_mask if not self.training else data.train_mask
        query_embeddings = node_embeddings[query_mask]
        
        # Generate parent candidate representations
        father_queries = self.father_predictor(query_embeddings)
        mother_queries = self.mother_predictor(query_embeddings)
        
        # Score all possible parents
        father_scores = self._compute_parent_scores(
            father_queries, node_embeddings, data, parent_type='father'
        )
        mother_scores = self._compute_parent_scores(
            mother_queries, node_embeddings, data, parent_type='mother'
        )
        
        return {
            'father_scores': father_scores,
            'mother_scores': mother_scores,
            'node_embeddings': node_embeddings
        }
    
    def _compute_parent_scores(self, 
                              queries: torch.Tensor,
                              all_embeddings: torch.Tensor,
                              data: Data,
                              parent_type: str) -> torch.Tensor:
        """Compute scores for all possible parents"""
        num_queries = queries.shape[0]
        num_candidates = all_embeddings.shape[0]
        
        # Expand for broadcasting
        queries_exp = queries.unsqueeze(1).expand(-1, num_candidates, -1)
        candidates_exp = all_embeddings.unsqueeze(0).expand(num_queries, -1, -1)
        
        # Compute compatibility scores
        scores = self.compatibility_scorer(queries_exp, candidates_exp).squeeze(-1)
        
        # Apply constraints
        scores = self._apply_constraints(scores, data, parent_type)
        
        return scores
    
    def _apply_constraints(self, 
                          scores: torch.Tensor,
                          data: Data,
                          parent_type: str) -> torch.Tensor:
        """Apply logical constraints to parent scores"""
        # Gender constraint
        if hasattr(data, 'gender'):
            if parent_type == 'father':
                # Mask out non-male candidates
                male_mask = (data.gender == 1)  # Assuming 1 = male
                scores = scores.masked_fill(~male_mask.unsqueeze(0), -float('inf'))
            else:  # mother
                # Mask out non-female candidates
                female_mask = (data.gender == 0)  # Assuming 0 = female
                scores = scores.masked_fill(~female_mask.unsqueeze(0), -float('inf'))
        
        # Generation constraint
        if hasattr(data, 'generation'):
            # Parents should be ~1 generation older
            gen_diff = data.generation.unsqueeze(0) - data.generation.unsqueeze(1)
            invalid_gen = (gen_diff < 0.5) | (gen_diff > 2.0)
            scores = scores.masked_fill(invalid_gen, -float('inf'))
        
        return scores
    
    def predict_parents(self, data: Data, top_k: int = 5) -> Dict[str, torch.Tensor]:
        """Predict top-k most likely parents for each individual"""
        with torch.no_grad():
            outputs = self.forward(data)
            
            # Get top-k candidates
            father_topk = torch.topk(outputs['father_scores'], k=top_k, dim=1)
            mother_topk = torch.topk(outputs['mother_scores'], k=top_k, dim=1)
            
            return {
                'father_indices': father_topk.indices,
                'father_scores': father_topk.values,
                'mother_indices': mother_topk.indices,
                'mother_scores': mother_topk.values
            }


class HeterogeneousFamilyGNN(nn.Module):
    """Heterogeneous GNN for family data with different node and edge types"""
    
    def __init__(self,
                 metadata: Tuple[List[str], List[Tuple[str, str, str]]],
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        
        self.metadata = metadata
        node_types, edge_types = metadata
        
        # Node type embeddings
        self.node_embeddings = nn.ModuleDict({
            node_type: nn.Linear(hidden_dim, hidden_dim)
            for node_type in node_types
        })
        
        # Heterogeneous GNN layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                edge_type: GATConv(
                    hidden_dim, 
                    hidden_dim // num_heads if i < num_layers - 1 else hidden_dim,
                    heads=num_heads if i < num_layers - 1 else 1,
                    concat=i < num_layers - 1,
                    dropout=dropout
                )
                for edge_type in edge_types
            })
            self.convs.append(conv)
        
        # Output projections for parent prediction
        self.parent_predictor = nn.ModuleDict({
            'father': nn.Linear(hidden_dim, hidden_dim),
            'mother': nn.Linear(hidden_dim, hidden_dim)
        })
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Initial node embeddings
        for node_type, x in x_dict.items():
            x_dict[node_type] = F.relu(self.node_embeddings[node_type](x))
        
        # Apply heterogeneous GNN layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Get individual node embeddings for parent prediction
        individual_embeddings = x_dict['individual']
        
        # Generate parent predictions
        father_pred = self.parent_predictor['father'](individual_embeddings)
        mother_pred = self.parent_predictor['mother'](individual_embeddings)
        
        return {
            'embeddings': x_dict,
            'father_pred': father_pred,
            'mother_pred': mother_pred
        }


class AttentionPooling(nn.Module):
    """Attention-based pooling for aggregating information"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Compute attention weights
        weights = self.attention(x)
        weights = F.softmax(weights, dim=0)
        
        # Apply attention pooling
        if batch is None:
            return (x * weights).sum(dim=0, keepdim=True)
        else:
            return global_mean_pool(x * weights, batch)