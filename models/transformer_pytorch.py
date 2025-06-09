"""
Transformer Models for ʻŌhanaGPT using PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import math
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration for Transformer models"""
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 1000
    vocab_size: int = 50000


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class GenerationalPositionalEncoding(nn.Module):
    """Special positional encoding that considers generational relationships"""
    
    def __init__(self, d_model: int, max_generations: int = 20):
        super().__init__()
        self.d_model = d_model
        self.generation_embedding = nn.Embedding(max_generations, d_model)
        self.temporal_embedding = nn.Linear(1, d_model // 2)
        
    def forward(self, x: torch.Tensor, generation_info: torch.Tensor, 
                birth_years: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Add generation embeddings
        gen_emb = self.generation_embedding(generation_info)
        x = x + gen_emb
        
        # Add temporal information if available
        if birth_years is not None:
            # Normalize birth years
            normalized_years = (birth_years - 1900) / 100.0
            temporal_emb = self.temporal_embedding(normalized_years.unsqueeze(-1))
            # Pad to full dimension
            temporal_emb = F.pad(temporal_emb, (0, self.d_model - temporal_emb.size(-1)))
            x = x + temporal_emb
        
        return x


class FamilyTransformerEncoder(nn.Module):
    """Transformer encoder for genealogical data"""
    
    def __init__(self, config: TransformerConfig, input_dim: int):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, config.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout)
        self.gen_pos_encoder = GenerationalPositionalEncoding(config.d_model)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            config.d_model,
            config.nhead,
            config.dim_feedforward,
            config.dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers,
            config.num_encoder_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)
    
    def forward(self, 
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                generation_info: Optional[torch.Tensor] = None,
                birth_years: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project input features
        src = self.input_projection(src)
        
        # Add positional encoding
        if generation_info is not None:
            src = self.gen_pos_encoder(src, generation_info, birth_years)
        else:
            src = self.pos_encoder(src)
        
        # Apply transformer encoder
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        output = self.layer_norm(output)
        
        return output


class ParentPredictionTransformer(nn.Module):
    """Transformer model for predicting missing parents"""
    
    def __init__(self, 
                 node_feature_dim: int,
                 config: Optional[TransformerConfig] = None):
        super().__init__()
        self.config = config or TransformerConfig()
        
        # Encoder for processing individuals
        self.encoder = FamilyTransformerEncoder(self.config, node_feature_dim)
        
        # Specialized decoders for parent prediction
        self.parent_decoder = nn.TransformerDecoder(
            TransformerDecoderLayer(
                self.config.d_model,
                self.config.nhead,
                self.config.dim_feedforward,
                self.config.dropout,
                batch_first=True
            ),
            num_layers=self.config.num_decoder_layers
        )
        
        # Parent query embeddings
        self.father_query = nn.Parameter(torch.randn(1, 1, self.config.d_model))
        self.mother_query = nn.Parameter(torch.randn(1, 1, self.config.d_model))
        
        # Output heads
        self.father_head = nn.Linear(self.config.d_model, self.config.d_model)
        self.mother_head = nn.Linear(self.config.d_model, self.config.d_model)
        
        # Compatibility scorer
        self.compatibility = nn.Bilinear(self.config.d_model, self.config.d_model, 1)
        
        # Constraint modules
        self.gender_constraint = nn.Linear(1, self.config.d_model)
        self.generation_constraint = nn.Linear(2, self.config.d_model)
    
    def forward(self, 
                individual_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                generation_info: Optional[torch.Tensor] = None,
                gender_info: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size = individual_features.size(0)
        
        # Encode all individuals
        encoded_individuals = self.encoder(
            individual_features, 
            src_mask=mask,
            generation_info=generation_info
        )
        
        # Prepare parent queries
        father_queries = self.father_query.expand(batch_size, -1, -1)
        mother_queries = self.mother_query.expand(batch_size, -1, -1)
        
        # Decode parent representations
        father_decoded = self.parent_decoder(
            father_queries,
            encoded_individuals,
            memory_key_padding_mask=mask
        )
        mother_decoded = self.parent_decoder(
            mother_queries,
            encoded_individuals,
            memory_key_padding_mask=mask
        )
        
        # Generate parent representations
        father_repr = self.father_head(father_decoded.squeeze(1))
        mother_repr = self.mother_head(mother_decoded.squeeze(1))
        
        # Score compatibility with all individuals
        father_scores = self._compute_compatibility_scores(
            father_repr, encoded_individuals, gender_info, generation_info, 'father'
        )
        mother_scores = self._compute_compatibility_scores(
            mother_repr, encoded_individuals, gender_info, generation_info, 'mother'
        )
        
        return {
            'father_scores': father_scores,
            'mother_scores': mother_scores,
            'encoded_individuals': encoded_individuals,
            'father_representation': father_repr,
            'mother_representation': mother_repr
        }
    
    def _compute_compatibility_scores(self,
                                    parent_repr: torch.Tensor,
                                    all_individuals: torch.Tensor,
                                    gender_info: Optional[torch.Tensor],
                                    generation_info: Optional[torch.Tensor],
                                    parent_type: str) -> torch.Tensor:
        """Compute compatibility scores between parent representation and all individuals"""
        batch_size, seq_len, d_model = all_individuals.shape
        
        # Expand parent representation for comparison
        parent_repr_exp = parent_repr.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Base compatibility scores
        scores = self.compatibility(parent_repr_exp, all_individuals).squeeze(-1)
        
        # Apply gender constraints if available
        if gender_info is not None:
            gender_mask = self._get_gender_mask(gender_info, parent_type)
            scores = scores.masked_fill(~gender_mask, -float('inf'))
        
        # Apply generation constraints if available
        if generation_info is not None:
            gen_penalty = self._compute_generation_penalty(generation_info)
            scores = scores - gen_penalty
        
        return scores
    
    def _get_gender_mask(self, gender_info: torch.Tensor, parent_type: str) -> torch.Tensor:
        """Create mask based on gender constraints"""
        if parent_type == 'father':
            return gender_info == 1  # Male
        else:
            return gender_info == 0  # Female
    
    def _compute_generation_penalty(self, generation_info: torch.Tensor) -> torch.Tensor:
        """Compute penalty based on generation differences"""
        # Assuming generation_info shape: [batch_size, seq_len]
        gen_diff = generation_info.unsqueeze(1) - generation_info.unsqueeze(2)
        
        # Ideal parent-child generation difference is around 1
        penalty = torch.abs(gen_diff - 1.0) * 10.0  # Scale penalty
        
        # Additional penalty for impossible relationships
        penalty = torch.where(gen_diff <= 0, torch.tensor(float('inf')), penalty)
        
        return penalty


class FamilySeq2SeqTransformer(nn.Module):
    """Sequence-to-sequence transformer for genealogical completion"""
    
    def __init__(self,
                 node_feature_dim: int,
                 config: Optional[TransformerConfig] = None):
        super().__init__()
        self.config = config or TransformerConfig()
        
        # Shared embeddings
        self.feature_projection = nn.Linear(node_feature_dim, self.config.d_model)
        
        # Encoder
        self.encoder = FamilyTransformerEncoder(self.config, node_feature_dim)
        
        # Decoder with cross-attention
        decoder_layer = TransformerDecoderLayer(
            self.config.d_model,
            self.config.nhead,
            self.config.dim_feedforward,
            self.config.dropout,
            batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, self.config.num_decoder_layers)
        
        # Special tokens
        self.start_token = nn.Parameter(torch.randn(1, 1, self.config.d_model))
        self.query_token = nn.Parameter(torch.randn(1, 1, self.config.d_model))
        
        # Output projection
        self.output_projection = nn.Linear(self.config.d_model, node_feature_dim)
        
        # Relationship type embeddings
        self.rel_embeddings = nn.Embedding(5, self.config.d_model)  # parent, child, spouse, sibling, unknown
    
    def forward(self,
                src_individuals: torch.Tensor,
                tgt_queries: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Encode source individuals
        memory = self.encoder(src_individuals, src_mask)
        
        # Prepare target sequence with start token
        batch_size = tgt_queries.size(0)
        start_tokens = self.start_token.expand(batch_size, -1, -1)
        tgt_sequence = torch.cat([start_tokens, self.feature_projection(tgt_queries)], dim=1)
        
        # Decode
        output = self.decoder(
            tgt_sequence,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_mask
        )
        
        # Project back to feature space
        output = self.output_projection(output[:, 1:, :])  # Remove start token
        
        return output
    
    def generate_parents(self,
                        individual_features: torch.Tensor,
                        max_length: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate parent predictions autoregressively"""
        batch_size = individual_features.size(0)
        device = individual_features.device
        
        # Encode individuals
        memory = self.encoder(individual_features)
        
        # Initialize with start token
        generated = self.start_token.expand(batch_size, -1, -1)
        
        # Generate father and mother sequentially
        parent_predictions = []
        
        for _ in range(max_length):  # Generate 2 parents
            # Decode next parent
            output = self.decoder(generated, memory)
            next_parent = output[:, -1:, :]
            
            # Store prediction
            parent_predictions.append(next_parent)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_parent], dim=1)
        
        # Split into father and mother predictions
        father_pred = parent_predictions[0]
        mother_pred = parent_predictions[1] if len(parent_predictions) > 1 else None
        
        return father_pred, mother_pred


class HierarchicalFamilyTransformer(nn.Module):
    """Hierarchical transformer that processes family trees level by level"""
    
    def __init__(self,
                 node_feature_dim: int,
                 config: Optional[TransformerConfig] = None,
                 max_depth: int = 5):
        super().__init__()
        self.config = config or TransformerConfig()
        self.max_depth = max_depth
        
        # Level-specific encoders
        self.level_encoders = nn.ModuleList([
            FamilyTransformerEncoder(self.config, node_feature_dim)
            for _ in range(max_depth)
        ])
        
        # Cross-level attention
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(
                self.config.d_model,
                self.config.nhead,
                dropout=self.config.dropout,
                batch_first=True
            )
            for _ in range(max_depth - 1)
        ])
        
        # Level embeddings
        self.level_embedding = nn.Embedding(max_depth, self.config.d_model)
        
        # Final aggregation
        self.aggregator = nn.TransformerEncoder(
            TransformerEncoderLayer(
                self.config.d_model,
                self.config.nhead,
                self.config.dim_feedforward,
                self.config.dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Parent prediction heads
        self.parent_predictor = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.dim_feedforward, self.config.d_model * 2)  # Father and mother
        )
    
    def forward(self,
                individuals_by_level: List[torch.Tensor],
                masks_by_level: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        if masks_by_level is None:
            masks_by_level = [None] * len(individuals_by_level)
        
        # Process each level
        level_outputs = []
        
        for level, (individuals, mask) in enumerate(zip(individuals_by_level, masks_by_level)):
            # Add level embedding
            level_emb = self.level_embedding(torch.tensor(level, device=individuals.device))
            
            # Encode current level
            encoded = self.level_encoders[level](individuals, mask)
            
            # Add cross-level attention if not the first level
            if level > 0 and level_outputs:
                attended, _ = self.cross_attention[level-1](
                    encoded,
                    level_outputs[-1],
                    level_outputs[-1],
                    key_padding_mask=masks_by_level[level-1]
                )
                encoded = encoded + attended
            
            level_outputs.append(encoded)
        
        # Aggregate all levels
        all_levels = torch.cat(level_outputs, dim=1)
        aggregated = self.aggregator(all_levels)
        
        # Predict parents
        parent_predictions = self.parent_predictor(aggregated)
        father_pred, mother_pred = parent_predictions.chunk(2, dim=-1)
        
        return {
            'level_encodings': level_outputs,
            'aggregated': aggregated,
            'father_prediction': father_pred,
            'mother_prediction': mother_pred
        }


class RelationalTransformer(nn.Module):
    """Transformer with explicit relational reasoning capabilities"""
    
    def __init__(self,
                 node_feature_dim: int,
                 num_relation_types: int = 5,
                 config: Optional[TransformerConfig] = None):
        super().__init__()
        self.config = config or TransformerConfig()
        self.num_relation_types = num_relation_types
        
        # Feature projection
        self.node_projection = nn.Linear(node_feature_dim, self.config.d_model)
        
        # Relation-aware transformer layers
        self.layers = nn.ModuleList([
            RelationalTransformerLayer(
                self.config.d_model,
                self.config.nhead,
                self.config.dim_feedforward,
                self.config.dropout,
                num_relation_types
            )
            for _ in range(self.config.num_encoder_layers)
        ])
        
        # Output heads
        self.parent_classifier = nn.Linear(self.config.d_model, 2)  # Binary: is parent or not
        self.relation_classifier = nn.Linear(self.config.d_model * 2, num_relation_types)
    
    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Project node features
        x = self.node_projection(node_features)
        
        # Apply relational transformer layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
        
        # Predict parent relationships
        parent_logits = self.parent_classifier(x)
        
        # Predict edge types for all pairs
        # This is simplified - in practice, you'd only compute for relevant pairs
        edge_features = torch.cat([
            x[edge_index[0]],
            x[edge_index[1]]
        ], dim=-1)
        relation_logits = self.relation_classifier(edge_features)
        
        return {
            'node_embeddings': x,
            'parent_logits': parent_logits,
            'relation_logits': relation_logits
        }


class RelationalTransformerLayer(nn.Module):
    """Single layer of relational transformer"""
    
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int,
                 dropout: float,
                 num_relation_types: int):
        super().__init__()
        
        # Relation-specific attention
        self.relation_attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_relation_types)
        ])
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        # Relation-specific attention
        attn_outputs = []
        for rel_type in range(len(self.relation_attention)):
            # Create attention mask for this relation type
            rel_mask = (edge_type == rel_type)
            
            if rel_mask.any():
                # Apply attention for this relation
                attn_out, _ = self.relation_attention[rel_type](x, x, x)
                attn_outputs.append(attn_out)
        
        # Aggregate attention outputs
        if attn_outputs:
            attn_output = sum(attn_outputs) / len(attn_outputs)
        else:
            attn_output = x
        
        # Add & norm
        x = self.norm1(x + self.dropout(attn_output))
        
        # FFN
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x