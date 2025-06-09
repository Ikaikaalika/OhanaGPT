"""
Transformer Models for ʻŌhanaGPT using MLX (Apple Silicon)
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math
from typing import Dict, Optional, Tuple, List, NamedTuple
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


class MultiHeadAttention(nn.Module):
    """Multi-head attention implementation for MLX"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = dropout
    
    def __call__(self, query: mx.array, key: mx.array, value: mx.array,
                 mask: Optional[mx.array] = None) -> mx.array:
        batch_size, seq_len, _ = query.shape
        
        # Linear transformations and reshape
        Q = self.w_q(query).reshape(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.w_k(key).reshape(batch_size, -1, self.num_heads, self.d_k)
        V = self.w_v(value).reshape(batch_size, -1, self.num_heads, self.d_k)
        
        # Transpose for attention computation
        Q = Q.transpose(0, 2, 1, 3)  # [batch, heads, seq, d_k]
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Compute attention scores
        scores = mx.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.reshape(batch_size, 1, 1, -1)
            scores = mx.where(mask, scores, -1e9)
        
        # Apply softmax
        attn_weights = mx.softmax(scores, axis=-1)
        
        # Apply dropout
        if self.training and self.dropout > 0:
            attn_weights = mx.dropout(attn_weights, self.dropout)
        
        # Apply attention to values
        attn_output = mx.matmul(attn_weights, V)
        
        # Transpose and reshape
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # Final linear transformation
        output = self.w_o(attn_output)
        
        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer for MLX"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = dropout
    
    def __call__(self, src: mx.array, src_mask: Optional[mx.array] = None) -> mx.array:
        # Self-attention
        attn_output = self.self_attn(src, src, src, src_mask)
        if self.training and self.dropout > 0:
            attn_output = mx.dropout(attn_output, self.dropout)
        src = self.norm1(src + attn_output)
        
        # Feed-forward network
        ff_output = self.linear2(mx.nn.relu(self.linear1(src)))
        if self.training and self.dropout > 0:
            ff_output = mx.dropout(ff_output, self.dropout)
        src = self.norm2(src + ff_output)
        
        return src


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers"""
    
    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = [encoder_layer for _ in range(num_layers)]
        self.num_layers = num_layers
    
    def __call__(self, src: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        output = src
        for layer in self.layers:
            output = layer(output, mask)
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models in MLX"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = mx.zeros((max_len, d_model))
        position = mx.arange(0, max_len).reshape(-1, 1).astype(mx.float32)
        
        div_term = mx.exp(mx.arange(0, d_model, 2).astype(mx.float32) * 
                         (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = mx.sin(position * div_term)
        pe[:, 1::2] = mx.cos(position * div_term)
        
        self.pe = pe[None, :, :]  # Add batch dimension
    
    def __call__(self, x: mx.array) -> mx.array:
        return x + self.pe[:, :x.shape[1], :]


class GenerationalPositionalEncoding(nn.Module):
    """Special positional encoding that considers generational relationships"""
    
    def __init__(self, d_model: int, max_generations: int = 20):
        super().__init__()
        self.d_model = d_model
        self.generation_embedding = nn.Embedding(max_generations, d_model)
        self.temporal_linear = nn.Linear(1, d_model // 2)
    
    def __call__(self, x: mx.array, generation_info: mx.array, 
                 birth_years: Optional[mx.array] = None) -> mx.array:
        # Add generation embeddings
        gen_emb = self.generation_embedding(generation_info)
        x = x + gen_emb
        
        # Add temporal information if available
        if birth_years is not None:
            # Normalize birth years
            normalized_years = (birth_years - 1900) / 100.0
            temporal_emb = self.temporal_linear(normalized_years[:, :, None])
            # Pad to full dimension
            padding = mx.zeros((temporal_emb.shape[0], temporal_emb.shape[1], 
                              self.d_model - temporal_emb.shape[2]))
            temporal_emb = mx.concatenate([temporal_emb, padding], axis=-1)
            x = x + temporal_emb
        
        return x


class FamilyTransformerEncoder_MLX(nn.Module):
    """Transformer encoder for genealogical data in MLX"""
    
    def __init__(self, config: TransformerConfig, input_dim: int):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, config.d_model)
        
        # Positional encodings
        self.pos_encoder = PositionalEncoding(config.d_model)
        self.gen_pos_encoder = GenerationalPositionalEncoding(config.d_model)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            config.d_model,
            config.nhead,
            config.dim_feedforward,
            config.dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, config.num_encoder_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)
    
    def __call__(self, 
                 src: mx.array,
                 src_mask: Optional[mx.array] = None,
                 generation_info: Optional[mx.array] = None,
                 birth_years: Optional[mx.array] = None) -> mx.array:
        # Project input features
        src = self.input_projection(src)
        
        # Add positional encoding
        if generation_info is not None:
            src = self.gen_pos_encoder(src, generation_info, birth_years)
        else:
            src = self.pos_encoder(src)
        
        # Apply transformer encoder
        output = self.transformer_encoder(src, src_mask)
        output = self.layer_norm(output)
        
        return output


class ParentPredictionTransformer_MLX(nn.Module):
    """Transformer model for predicting missing parents in MLX"""
    
    def __init__(self, 
                 node_feature_dim: int,
                 config: Optional[TransformerConfig] = None):
        super().__init__()
        self.config = config or TransformerConfig()
        
        # Encoder for processing individuals
        self.encoder = FamilyTransformerEncoder_MLX(self.config, node_feature_dim)
        
        # Parent query embeddings (learnable)
        self.father_query = mx.random.normal((1, 1, self.config.d_model))
        self.mother_query = mx.random.normal((1, 1, self.config.d_model))
        
        # Cross-attention for parent prediction
        self.parent_attention = MultiHeadAttention(
            self.config.d_model, 
            self.config.nhead, 
            self.config.dropout
        )
        
        # Output heads
        self.father_head = nn.Linear(self.config.d_model, self.config.d_model)
        self.mother_head = nn.Linear(self.config.d_model, self.config.d_model)
        
        # Scoring layers
        self.score_projection = nn.Linear(self.config.d_model * 2, 1)
    
    def __call__(self, 
                 individual_features: mx.array,
                 mask: Optional[mx.array] = None,
                 generation_info: Optional[mx.array] = None,
                 gender_info: Optional[mx.array] = None) -> Dict[str, mx.array]:
        batch_size = individual_features.shape[0]
        
        # Encode all individuals
        encoded_individuals = self.encoder(
            individual_features, 
            src_mask=mask,
            generation_info=generation_info
        )
        
        # Prepare parent queries
        father_queries = mx.broadcast_to(self.father_query, (batch_size, 1, self.config.d_model))
        mother_queries = mx.broadcast_to(self.mother_query, (batch_size, 1, self.config.d_model))
        
        # Cross-attention to find parents
        father_attended = self.parent_attention(
            father_queries, 
            encoded_individuals, 
            encoded_individuals,
            mask
        )
        mother_attended = self.parent_attention(
            mother_queries,
            encoded_individuals,
            encoded_individuals,
            mask
        )
        
        # Generate parent representations
        father_repr = self.father_head(father_attended.squeeze(1))
        mother_repr = self.mother_head(mother_attended.squeeze(1))
        
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
                                    parent_repr: mx.array,
                                    all_individuals: mx.array,
                                    gender_info: Optional[mx.array],
                                    generation_info: Optional[mx.array],
                                    parent_type: str) -> mx.array:
        """Compute compatibility scores between parent representation and all individuals"""
        batch_size, seq_len, d_model = all_individuals.shape
        
        # Expand parent representation
        parent_repr_exp = parent_repr[:, None, :].broadcast_to((batch_size, seq_len, d_model))
        
        # Concatenate representations
        combined = mx.concatenate([parent_repr_exp, all_individuals], axis=-1)
        
        # Score compatibility
        scores = self.score_projection(combined).squeeze(-1)
        
        # Apply constraints
        if gender_info is not None:
            gender_mask = self._get_gender_mask(gender_info, parent_type)
            scores = mx.where(gender_mask, scores, -float('inf'))
        
        if generation_info is not None:
            gen_penalty = self._compute_generation_penalty(generation_info)
            scores = scores - gen_penalty
        
        return scores
    
    def _get_gender_mask(self, gender_info: mx.array, parent_type: str) -> mx.array:
        """Create mask based on gender constraints"""
        if parent_type == 'father':
            return gender_info == 1  # Male
        else:
            return gender_info == 0  # Female
    
    def _compute_generation_penalty(self, generation_info: mx.array) -> mx.array:
        """Compute penalty based on generation differences"""
        # Expand dimensions for broadcasting
        gen_diff = generation_info[:, None] - generation_info[None, :]
        
        # Ideal parent-child generation difference is around 1
        penalty = mx.abs(gen_diff - 1.0) * 10.0
        
        # Additional penalty for impossible relationships
        penalty = mx.where(gen_diff <= 0, float('inf'), penalty)
        
        return penalty
    
    def predict_parents(self, 
                       individual_features: mx.array,
                       top_k: int = 5,
                       **kwargs) -> Dict[str, mx.array]:
        """Predict top-k most likely parents"""
        outputs = self(individual_features, **kwargs)
        
        # Get top-k candidates
        father_topk = mx.topk(outputs['father_scores'], k=top_k, axis=1)
        mother_topk = mx.topk(outputs['mother_scores'], k=top_k, axis=1)
        
        return {
            'father_indices': father_topk[1],
            'father_scores': father_topk[0],
            'mother_indices': mother_topk[1],
            'mother_scores': mother_topk[0]
        }


class HierarchicalTransformer_MLX(nn.Module):
    """Hierarchical transformer for processing family trees level by level"""
    
    def __init__(self,
                 node_feature_dim: int,
                 config: Optional[TransformerConfig] = None,
                 max_depth: int = 5):
        super().__init__()
        self.config = config or TransformerConfig()
        self.max_depth = max_depth
        
        # Level-specific encoders
        self.level_encoders = [
            FamilyTransformerEncoder_MLX(self.config, node_feature_dim)
            for _ in range(max_depth)
        ]
        
        # Cross-level attention
        self.cross_attention = [
            MultiHeadAttention(
                self.config.d_model,
                self.config.nhead,
                self.config.dropout
            )
            for _ in range(max_depth - 1)
        ]
        
        # Level embeddings
        self.level_embedding = nn.Embedding(max_depth, self.config.d_model)
        
        # Final aggregation
        self.aggregator = TransformerEncoder(
            TransformerEncoderLayer(
                self.config.d_model,
                self.config.nhead,
                self.config.dim_feedforward,
                self.config.dropout
            ),
            num_layers=2
        )
        
        # Parent prediction head
        self.parent_predictor = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.config.dim_feedforward, self.config.d_model * 2)
        )
    
    def __call__(self,
                 individuals_by_level: List[mx.array],
                 masks_by_level: Optional[List[mx.array]] = None) -> Dict[str, mx.array]:
        if masks_by_level is None:
            masks_by_level = [None] * len(individuals_by_level)
        
        # Process each level
        level_outputs = []
        
        for level, (individuals, mask) in enumerate(zip(individuals_by_level, masks_by_level)):
            # Add level embedding
            level_emb = self.level_embedding(mx.array([level]))
            
            # Encode current level
            encoded = self.level_encoders[level](individuals, mask)
            
            # Add cross-level attention if not the first level
            if level > 0 and level_outputs:
                attended = self.cross_attention[level-1](
                    encoded,
                    level_outputs[-1],
                    level_outputs[-1],
                    masks_by_level[level-1]
                )
                encoded = encoded + attended
            
            level_outputs.append(encoded)
        
        # Aggregate all levels
        all_levels = mx.concatenate(level_outputs, axis=1)
        aggregated = self.aggregator(all_levels)
        
        # Predict parents
        parent_predictions = self.parent_predictor(aggregated)
        d_model = self.config.d_model
        father_pred = parent_predictions[:, :, :d_model]
        mother_pred = parent_predictions[:, :, d_model:]
        
        return {
            'level_encodings': level_outputs,
            'aggregated': aggregated,
            'father_prediction': father_pred,
            'mother_prediction': mother_pred
        }