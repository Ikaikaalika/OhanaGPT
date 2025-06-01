import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlx
import mlx.optimizers as mlx_optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)

class IncrementalTrainer:
    """Handle incremental training with deduplication."""
    
    def __init__(self, model_type: str = 'gnn', framework: str = 'pytorch',
                 config: Optional[Dict] = None):
        self.model_type = model_type
        self.framework = framework
        self.config = config or {}
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.training_history = []
        self.seen_individuals = set()
        self.deduplication_index = {}
        
        # Model save path
        self.model_path = Path(self.config.get('model_dir', './models'))
        self.model_path.mkdir(exist_ok=True)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model based on type and framework."""
        if self.framework == 'pytorch':
            self._init_pytorch_model()
        elif self.framework == 'mlx':
            self._init_mlx_model()
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
    
    def _init_pytorch_model(self):
        """Initialize PyTorch model."""
        from ..models.pytorch_models import GenealogyGNN, TransformerGenealogyModel
        
        input_dim = self.config.get('input_dim', 7)
        hidden_dim = self.config.get('hidden_dim', 256)
        output_dim = self.config.get('output_dim', 256)
        num_layers = self.config.get('num_layers', 4)
        dropout = self.config.get('dropout', 0.1)
        
        if self.model_type == 'gnn':
            self.model = GenealogyGNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        elif self.model_type == 'transformer':
            self.model = TransformerGenealogyModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=8,
                num_layers=num_layers,
                dropout=dropout
            )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001)
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5
        )

    def _init_mlx_model(self):
        """Initialize MLX model."""
        from ..models.mlx_models import MLXGenealogyGNN, MLXTransformerGenealogy
        
        input_dim = self.config.get('input_dim', 7)
        hidden_dim = self.config.get('hidden_dim', 256)
        output_dim = self.config.get('output_dim', 256)
        num_layers = self.config.get('num_layers', 4)
        dropout = self.config.get('dropout', 0.1)
        
        if self.model_type == 'gnn':
            self.model = MLXGenealogyGNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        elif self.model_type == 'transformer':
            self.model = MLXTransformerGenealogy(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=8,
                num_layers=num_layers,
                dropout=dropout
            )
        
        # Initialize MLX optimizer
        self.optimizer = mlx_optim.Adam(
            learning_rate=self.config.get('learning_rate', 0.001)
        )

    def train_on_new_data(self, graph_data: Dict, epochs: int = 10) -> Dict:
        """Train incrementally on new GEDCOM data."""
        logger.info(f"Starting incremental training on new data")
        
        # Check for duplicates
        new_individuals = self._filter_duplicates(graph_data)
        if not new_individuals:
            logger.info("No new individuals to train on")
            return {'status': 'skipped', 'reason': 'no_new_data'}
        
        # Prepare training data
        train_data = self._prepare_training_data(graph_data, new_individuals)
        
        # Train model
        if self.framework == 'pytorch':
            metrics = self._train_pytorch(train_data, epochs)
        else:
            metrics = self._train_mlx(train_data, epochs)
        
        # Update seen individuals
        self.seen_individuals.update(new_individuals)
        
        # Save model checkpoint
        self._save_checkpoint()
        
        # Record training history
        self.training_history.append({
            'timestamp': datetime.now(),
            'new_individuals': len(new_individuals),
            'total_individuals': len(self.seen_individuals),
            'metrics': metrics
        })
        
        return {
            'status': 'success',
            'new_individuals': len(new_individuals),
            'metrics': metrics
        }

    def _filter_duplicates(self, graph_data: Dict) -> set:
        """Filter out duplicate individuals based on similarity."""
        new_individuals = set()
        
        for ind_id, individual in graph_data['individuals'].items():
            # Create fingerprint for deduplication
            fingerprint = self._create_individual_fingerprint(individual)
            
            # Check similarity with existing individuals
            is_duplicate = False
            for existing_fp in self.deduplication_index.values():
                similarity = self._compute_similarity(fingerprint, existing_fp)
                if similarity > self.config.get('deduplication_threshold', 0.95):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                new_individuals.add(ind_id)
                self.deduplication_index[ind_id] = fingerprint
        
        return new_individuals

    def _create_individual_fingerprint(self, individual: 'Individual') -> Dict:
        """Create a fingerprint for deduplication."""
        return {
            'name': individual.name or '',
            'birth_year': individual.birth_date.year if individual.birth_date else None,
            'death_year': individual.death_date.year if individual.death_date else None,
            'gender': individual.gender,
            'num_spouses': len(individual.family_spouse),
            'has_parents': bool(individual.family_child)
        }

    def _compute_similarity(self, fp1: Dict, fp2: Dict) -> float:
        """Compute similarity between two fingerprints."""
        score = 0.0
        total = 0.0
        
        # Name similarity (using simple approach, can use more advanced methods)
        if fp1['name'] and fp2['name']:
            name_sim = 1.0 if fp1['name'].lower() == fp2['name'].lower() else 0.0
            score += name_sim * 0.4
            total += 0.4
        
        # Birth year similarity
        if fp1['birth_year'] and fp2['birth_year']:
            year_diff = abs(fp1['birth_year'] - fp2['birth_year'])
            year_sim = max(0, 1 - year_diff / 10.0)
            score += year_sim * 0.3
            total += 0.3
        
        # Other attributes
        if fp1['gender'] == fp2['gender']:
            score += 0.1
        total += 0.1
        
        if fp1['num_spouses'] == fp2['num_spouses']:
            score += 0.1
        total += 0.1
        
        if fp1['has_parents'] == fp2['has_parents']:
            score += 0.1
        total += 0.1
        
        return score / total if total > 0 else 0.0

    def _prepare_training_data(self, graph_data: Dict, new_individuals: set) -> Dict:
        """Prepare training data from graph."""
        # This would convert the graph data into training batches
        # Including positive and negative examples for parent prediction
        # Implementation depends on specific training approach
        pass

    def _train_pytorch(self, train_data: Dict, epochs: int) -> Dict:
        """Train PyTorch model."""
        self.model.train()
        total_loss = 0.0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Training loop would go here
            # This is a simplified version
            for batch in train_data['batches']:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    batch['node_features'],
                    batch['edge_index'],
                    batch['edge_types']
                )
                
                # Compute loss (simplified)
                loss = self._compute_loss(outputs, batch['labels'])
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_data['batches'])
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Update scheduler
            self.scheduler.step(avg_loss)
        
        return {'final_loss': avg_loss}

    def _train_mlx(self, train_data: Dict, epochs: int) -> Dict:
        """Train MLX model."""
        # MLX training implementation
        # Similar to PyTorch but using MLX operations
        pass

    def _compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute training loss."""
        # Implement appropriate loss function for parent prediction
        # Could be contrastive loss, cross-entropy, etc.
        criterion = nn.CrossEntropyLoss()
        return criterion(outputs, labels)

    def _save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint = {
            'model_state': self.model.state_dict() if self.framework == 'pytorch' else self.model,
            'optimizer_state': self.optimizer.state_dict() if self.framework == 'pytorch' else None,
            'training_history': self.training_history,
            'seen_individuals': self.seen_individuals,
            'deduplication_index': self.deduplication_index,
            'config': self.config
        }
        
        checkpoint_path = self.model_path / f"{self.model_type}_{self.framework}_checkpoint.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Optional[Path] = None):
        """Load model checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.model_path / f"{self.model_type}_{self.framework}_checkpoint.pkl"
        
        if not checkpoint_path.exists():
            logger.warning(f"No checkpoint found at {checkpoint_path}")
            return False
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        if self.framework == 'pytorch':
            self.model.load_state_dict(checkpoint['model_state'])
            if checkpoint['optimizer_state']:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        else:
            self.model = checkpoint['model_state']
        
        self.training_history = checkpoint['training_history']
        self.seen_individuals = checkpoint['seen_individuals']
        self.deduplication_index = checkpoint['deduplication_index']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return True

    def predict_parents(self, graph_data: Dict, individual_id: str) -> List[Tuple[str, float]]:
        """Predict potential parents for an individual."""
        if self.framework == 'pytorch':
            return self._predict_pytorch(graph_data, individual_id)
        else:
            return self._predict_mlx(graph_data, individual_id)

    def _predict_pytorch(self, graph_data: Dict, individual_id: str) -> List[Tuple[str, float]]:
        """Predict using PyTorch model."""
        self.model.eval()
        
        with torch.no_grad():
            # Convert graph to PyTorch format
            x, edge_index, edge_types = graph_data['pytorch_data']
            
            # Get individual index
            ind_idx = graph_data['node_to_idx'][individual_id]
            
            # Get predictions
            parent_probs = self.model.predict_parents(x, edge_index, edge_types, ind_idx)
            
            # Get top candidates
            top_k = 10
            top_probs, top_indices = torch.topk(parent_probs, top_k)
            
            # Convert to individual IDs
            results = []
            for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
                if idx != ind_idx:  # Exclude self
                    parent_id = graph_data['idx_to_node'][idx]
                    results.append((parent_id, prob))
            
            return results

    def _predict_mlx(self, graph_data: Dict, individual_id: str) -> List[Tuple[str, float]]:
        """Predict using MLX model."""
        # Similar to PyTorch but using MLX operations
        pass