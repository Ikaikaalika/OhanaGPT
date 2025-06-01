import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Tuple, Optional

class GenealogyGNN(nn.Module):
    """Graph Neural Network for genealogy parent prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            else:
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
        
        # Edge type embedding
        self.edge_embedding = nn.Embedding(3, hidden_dim)
        
        # Output layers
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_type: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the GNN."""
        # Project input features
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply GNN layers
        for i, layer in enumerate(self.gnn_layers):
            x_res = x
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i > 0:  # Skip connection
                x = x + x_res
        
        return x
    
    def predict_parents(self, x: torch.Tensor, edge_index: torch.Tensor,
                       edge_type: torch.Tensor, candidate_idx: int) -> torch.Tensor:
        """Predict potential parents for a given individual."""
        # Get node embeddings
        node_embeddings = self.forward(x, edge_index, edge_type)
        
        # Get candidate embedding
        candidate_embedding = node_embeddings[candidate_idx]
        
        # Compute similarity scores with all other nodes
        scores = torch.matmul(node_embeddings, candidate_embedding)
        
        # Apply softmax to get probabilities
        parent_probs = F.softmax(scores, dim=0)
        
        return parent_probs

class TransformerGenealogyModel(nn.Module):
    """Transformer-based model for genealogy parent prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 8,
                 num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the transformer."""
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Output projection
        x = self.output_proj(x)
        
        return x
    
    def predict_parents(self, x: torch.Tensor, candidate_idx: int) -> torch.Tensor:
        """Predict potential parents using transformer embeddings."""
        # Get embeddings
        embeddings = self.forward(x.unsqueeze(0))
        embeddings = embeddings.squeeze(0)
        
        # Get candidate embedding
        candidate_embedding = embeddings[candidate_idx]
        
        # Compute similarity scores
        scores = torch.matmul(embeddings, candidate_embedding)
        parent_probs = F.softmax(scores, dim=0)
        
        return parent_probs

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)