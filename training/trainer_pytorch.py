"""
PyTorch Training Module for ʻŌhanaGPT
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeometricDataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import wandb
import os
from pathlib import Path
import json
from datetime import datetime
import logging

from ..models.gnn_pytorch import ParentPredictionGNN
from ..models.transformer_pytorch import ParentPredictionTransformer
from ..utils.metrics import compute_metrics, evaluate_predictions


class FamilyDataset(Dataset):
    """Dataset for family tree data"""
    
    def __init__(self, 
                 graph_data_list: List[Data],
                 transform=None):
        self.graphs = graph_data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        if self.transform:
            graph = self.transform(graph)
        return graph


class ParentPredictionLoss(nn.Module):
    """Custom loss function for parent prediction"""
    
    def __init__(self, 
                 alpha: float = 1.0,
                 beta: float = 0.5,
                 gamma: float = 0.1):
        super().__init__()
        self.alpha = alpha  # Weight for parent prediction loss
        self.beta = beta   # Weight for generation constraint
        self.gamma = gamma # Weight for consistency loss
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: torch.Tensor,
                mask: torch.Tensor,
                data: Optional[Data] = None) -> Dict[str, torch.Tensor]:
        device = predictions['father_scores'].device
        
        # Extract predictions
        father_scores = predictions['father_scores']
        mother_scores = predictions['mother_scores']
        
        # Get target indices
        father_targets = targets[:, 0]
        mother_targets = targets[:, 1]
        
        # Compute cross-entropy loss for parent predictions
        father_loss = nn.functional.cross_entropy(
            father_scores[mask],
            father_targets[mask],
            ignore_index=-1
        )
        
        mother_loss = nn.functional.cross_entropy(
            mother_scores[mask],
            mother_targets[mask],
            ignore_index=-1
        )
        
        parent_loss = (father_loss + mother_loss) / 2
        
        # Generation constraint loss
        gen_loss = torch.tensor(0.0, device=device)
        if data is not None and hasattr(data, 'generation'):
            # Ensure predicted parents are older
            father_preds = torch.argmax(father_scores, dim=1)
            mother_preds = torch.argmax(mother_scores, dim=1)
            
            child_gen = data.generation[mask]
            father_gen = data.generation[father_preds[mask]]
            mother_gen = data.generation[mother_preds[mask]]
            
            # Penalize if parents are not ~1 generation older
            gen_diff_father = child_gen - father_gen
            gen_diff_mother = child_gen - mother_gen
            
            gen_loss = (
                torch.abs(gen_diff_father - 1.0).mean() +
                torch.abs(gen_diff_mother - 1.0).mean()
            ) / 2
        
        # Consistency loss - parents should be spouses
        consistency_loss = torch.tensor(0.0, device=device)
        if 'node_embeddings' in predictions:
            embeddings = predictions['node_embeddings']
            
            # Get predicted parent embeddings
            father_preds = torch.argmax(father_scores, dim=1)
            mother_preds = torch.argmax(mother_scores, dim=1)
            
            father_emb = embeddings[father_preds[mask]]
            mother_emb = embeddings[mother_preds[mask]]
            
            # Parents should have similar embeddings (often spouses)
            consistency_loss = nn.functional.mse_loss(father_emb, mother_emb)
        
        # Total loss
        total_loss = (
            self.alpha * parent_loss +
            self.beta * gen_loss +
            self.gamma * consistency_loss
        )
        
        return {
            'total_loss': total_loss,
            'parent_loss': parent_loss,
            'generation_loss': gen_loss,
            'consistency_loss': consistency_loss,
            'father_loss': father_loss,
            'mother_loss': mother_loss
        }


class GNNTrainer:
    """Trainer for Graph Neural Network models"""
    
    def __init__(self,
                 model: ParentPredictionGNN,
                 config: Dict[str, Any],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.get('scheduler_patience', 10),
            verbose=True
        )
        
        # Loss function
        self.criterion = ParentPredictionLoss(
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 0.5),
            gamma=config.get('gamma', 0.1)
        )
        
        # Logging
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project="ohana-gpt",
                config=config,
                name=f"gnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Metrics tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        loss_components = {
            'parent_loss': 0,
            'generation_loss': 0,
            'consistency_loss': 0,
            'father_loss': 0,
            'mother_loss': 0
        }
        
        for batch in tqdm(train_loader, desc="Training"):
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch)
            
            # Compute loss
            losses = self.criterion(
                predictions,
                batch.y,
                batch.train_mask,
                batch
            )
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.get('grad_clip', 1.0)
            )
            
            self.optimizer.step()
            
            # Track losses
            total_loss += losses['total_loss'].item()
            for key in loss_components:
                if key in losses:
                    loss_components[key] += losses[key].item()
        
        # Average losses
        num_batches = len(train_loader)
        avg_losses = {
            'train_loss': total_loss / num_batches,
            **{f'train_{k}': v / num_batches for k, v in loss_components.items()}
        }
        
        return avg_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = batch.to(self.device)
                
                # Forward pass
                predictions = self.model(batch)
                
                # Compute loss
                losses = self.criterion(
                    predictions,
                    batch.y,
                    batch.val_mask,
                    batch
                )
                
                total_loss += losses['total_loss'].item()
                
                # Store predictions for metrics
                father_preds = torch.argmax(predictions['father_scores'], dim=1)
                mother_preds = torch.argmax(predictions['mother_scores'], dim=1)
                
                all_predictions.append(torch.stack([father_preds, mother_preds], dim=1))
                all_targets.append(batch.y)
                all_masks.append(batch.val_mask)
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        metrics = compute_metrics(all_predictions[all_masks], all_targets[all_masks])
        
        avg_loss = total_loss / len(val_loader)
        
        return {
            'val_loss': avg_loss,
            **{f'val_{k}': v for k, v in metrics.items()}
        }
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 100):
        """Full training loop"""
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log metrics
            all_metrics = {**train_losses, **val_metrics}
            self._log_metrics(all_metrics, epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['val_loss'])
            
            # Early stopping check
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics)
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config.get('early_stopping_patience', 20):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        self._load_best_checkpoint()
    
    def _log_metrics(self, metrics: Dict[str, float], epoch: int):
        """Log metrics to console and wandb"""
        # Console logging
        log_str = f"Epoch {epoch+1}: "
        log_str += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(log_str)
        
        # Wandb logging
        if self.use_wandb:
            wandb.log(metrics, step=epoch)
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}_loss_{metrics["val_loss"]:.4f}.pt'
        torch.save(checkpoint, path)
        
        # Also save as best
        best_path = self.checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        
        print(f"Saved checkpoint: {path}")
    
    def _load_best_checkpoint(self):
        """Load the best model checkpoint"""
        best_path = self.checkpoint_dir / 'best_model.pt'
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")


class TransformerTrainer:
    """Trainer for Transformer models"""
    
    def __init__(self,
                 model: ParentPredictionTransformer,
                 config: Dict[str, Any],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Similar setup to GNNTrainer but adapted for transformer inputs
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Warmup scheduler
        self.warmup_steps = config.get('warmup_steps', 1000)
        self.scheduler = self._get_linear_schedule_with_warmup(
            self.optimizer,
            self.warmup_steps,
            config.get('total_steps', 10000)
        )
        
        self.criterion = ParentPredictionLoss(
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 0.5),
            gamma=config.get('gamma', 0.1)
        )
        
        # Other setup similar to GNNTrainer
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project="ohana-gpt",
                config=config,
                name=f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def _get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        """Create a schedule with linear warmup and linear decay"""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / 
                float(max(1, num_training_steps - num_warmup_steps))
            )
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train transformer for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training Transformer"):
            # Prepare batch data
            individual_features = batch['features'].to(self.device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(self.device)
            
            generation_info = batch.get('generation_info', None)
            if generation_info is not None:
                generation_info = generation_info.to(self.device)
            
            gender_info = batch.get('gender_info', None)
            if gender_info is not None:
                gender_info = gender_info.to(self.device)
            
            targets = batch['targets'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(
                individual_features,
                mask=mask,
                generation_info=generation_info,
                gender_info=gender_info
            )
            
            # Compute loss
            train_mask = batch['train_mask'].to(self.device)
            losses = self.criterion(predictions, targets, train_mask)
            
            # Backward pass
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += losses['total_loss'].item()
        
        return {'train_loss': total_loss / len(train_loader)}


class IncrementalTrainer:
    """Trainer that supports incremental learning with new GEDCOM uploads"""
    
    def __init__(self,
                 model: nn.Module,
                 base_trainer: Union[GNNTrainer, TransformerTrainer],
                 config: Dict[str, Any]):
        self.model = model
        self.base_trainer = base_trainer
        self.config = config
        
        # Experience replay buffer
        self.replay_buffer = []
        self.buffer_size = config.get('replay_buffer_size', 1000)
        
        # Regularization for preventing catastrophic forgetting
        self.ewc_lambda = config.get('ewc_lambda', 0.1)
        self.fisher_dict = {}
        self.optimal_params_dict = {}
    
    def update_with_new_data(self, 
                            new_data_loader: DataLoader,
                            old_data_sample_loader: Optional[DataLoader] = None):
        """Update model with new GEDCOM data while preserving old knowledge"""
        
        # Compute Fisher Information Matrix on old data if using EWC
        if self.ewc_lambda > 0 and old_data_sample_loader is not None:
            self._compute_fisher_information(old_data_sample_loader)
        
        # Train on combined old (sampled) and new data
        if old_data_sample_loader is not None:
            combined_loader = self._combine_dataloaders(
                old_data_sample_loader, 
                new_data_loader
            )
        else:
            combined_loader = new_data_loader
        
        # Modified training with EWC regularization
        self._train_with_ewc(combined_loader)
        
        # Update replay buffer
        self._update_replay_buffer(new_data_loader)
    
    def _compute_fisher_information(self, data_loader: DataLoader):
        """Compute Fisher Information Matrix for Elastic Weight Consolidation"""
        self.model.eval()
        
        # Store current optimal parameters
        for name, param in self.model.named_parameters():
            self.optimal_params_dict[name] = param.data.clone()
        
        # Initialize Fisher dict
        for name, param in self.model.named_parameters():
            self.fisher_dict[name] = torch.zeros_like(param)
        
        # Accumulate gradients
        for batch in data_loader:
            if isinstance(batch, Data):
                batch = batch.to(self.base_trainer.device)
                outputs = self.model(batch)
                
                # Use negative log-likelihood as loss
                loss = -outputs['father_scores'].mean() - outputs['mother_scores'].mean()
            else:
                # Handle transformer batch format
                loss = self._compute_transformer_fisher_loss(batch)
            
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_dict[name] += param.grad.data ** 2
        
        # Normalize
        for name in self.fisher_dict:
            self.fisher_dict[name] /= len(data_loader)
    
    def _train_with_ewc(self, data_loader: DataLoader):
        """Train with Elastic Weight Consolidation regularization"""
        original_criterion = self.base_trainer.criterion
        
        def ewc_loss_wrapper(predictions, targets, mask, data=None):
            # Original loss
            losses = original_criterion(predictions, targets, mask, data)
            
            # EWC regularization
            ewc_loss = 0
            for name, param in self.model.named_parameters():
                if name in self.fisher_dict:
                    ewc_loss += (self.fisher_dict[name] * 
                               (param - self.optimal_params_dict[name]) ** 2).sum()
            
            losses['ewc_loss'] = self.ewc_lambda * ewc_loss
            losses['total_loss'] += losses['ewc_loss']
            
            return losses
        
        # Temporarily replace criterion
        self.base_trainer.criterion = ewc_loss_wrapper
        
        # Train
        self.base_trainer.train_epoch(data_loader)
        
        # Restore original criterion
        self.base_trainer.criterion = original_criterion
    
    def _update_replay_buffer(self, new_data_loader: DataLoader):
        """Update replay buffer with new samples"""
        for batch in new_data_loader:
            if len(self.replay_buffer) >= self.buffer_size:
                # Remove oldest samples
                self.replay_buffer.pop(0)
            self.replay_buffer.append(batch)
    
    def _combine_dataloaders(self, 
                           old_loader: DataLoader, 
                           new_loader: DataLoader) -> DataLoader:
        """Combine old and new data loaders"""
        # Simple approach: concatenate datasets
        combined_data = []
        
        for batch in old_loader:
            combined_data.append(batch)
        
        for batch in new_loader:
            combined_data.append(batch)
        
        # Create new dataset and loader
        if isinstance(combined_data[0], Data):
            combined_loader = GeometricDataLoader(
                combined_data,
                batch_size=self.config.get('batch_size', 32),
                shuffle=True
            )
        else:
            # Handle other data formats
            combined_loader = DataLoader(
                combined_data,
                batch_size=self.config.get('batch_size', 32),
                shuffle=True
            )
        
        return combined_loader