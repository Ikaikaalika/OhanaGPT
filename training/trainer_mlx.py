"""
MLX Training Module for ʻŌhanaGPT (Apple Silicon Optimized)
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Iterator
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime
import pickle

from ..models.gnn_mlx import ParentPredictionGNN_MLX, GraphData
from ..models.transformer_mlx import ParentPredictionTransformer_MLX
from ..utils.metrics import compute_metrics_mlx


class ParentPredictionLoss_MLX:
    """Custom loss function for parent prediction in MLX"""
    
    def __init__(self, 
                 alpha: float = 1.0,
                 beta: float = 0.5,
                 gamma: float = 0.1):
        self.alpha = alpha  # Weight for parent prediction loss
        self.beta = beta   # Weight for generation constraint
        self.gamma = gamma # Weight for consistency loss
    
    def __call__(self, 
                 predictions: Dict[str, mx.array],
                 targets: mx.array,
                 mask: mx.array,
                 data: Optional[GraphData] = None) -> Dict[str, mx.array]:
        
        # Extract predictions
        father_scores = predictions['father_scores']
        mother_scores = predictions['mother_scores']
        
        # Get masked indices
        masked_indices = mx.where(mask)[0]
        
        # Get target indices
        father_targets = targets[masked_indices, 0]
        mother_targets = targets[masked_indices, 1]
        
        # Compute cross-entropy loss for parent predictions
        father_loss = self._cross_entropy(
            father_scores[masked_indices],
            father_targets
        )
        
        mother_loss = self._cross_entropy(
            mother_scores[masked_indices],
            mother_targets
        )
        
        parent_loss = (father_loss + mother_loss) / 2
        
        # Generation constraint loss
        gen_loss = mx.array(0.0)
        if data is not None and hasattr(data, 'generation') and data.generation is not None:
            # Get predictions
            father_preds = mx.argmax(father_scores, axis=1)
            mother_preds = mx.argmax(mother_scores, axis=1)
            
            # Get generations
            child_gen = data.generation[masked_indices]
            father_gen = data.generation[father_preds[masked_indices]]
            mother_gen = data.generation[mother_preds[masked_indices]]
            
            # Compute generation differences
            gen_diff_father = child_gen - father_gen
            gen_diff_mother = child_gen - mother_gen
            
            # Penalize if not ~1 generation older
            gen_loss = (
                mx.mean(mx.abs(gen_diff_father - 1.0)) +
                mx.mean(mx.abs(gen_diff_mother - 1.0))
            ) / 2
        
        # Consistency loss
        consistency_loss = mx.array(0.0)
        if 'node_embeddings' in predictions:
            embeddings = predictions['node_embeddings']
            
            # Get predicted parent embeddings
            father_preds = mx.argmax(father_scores, axis=1)
            mother_preds = mx.argmax(mother_scores, axis=1)
            
            father_emb = embeddings[father_preds[masked_indices]]
            mother_emb = embeddings[mother_preds[masked_indices]]
            
            # MSE loss between parent embeddings
            consistency_loss = mx.mean((father_emb - mother_emb) ** 2)
        
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
    
    def _cross_entropy(self, logits: mx.array, targets: mx.array) -> mx.array:
        """Compute cross-entropy loss"""
        # Mask out invalid targets
        valid_mask = targets >= 0
        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]
        
        if valid_targets.size == 0:
            return mx.array(0.0)
        
        # Compute log softmax
        log_probs = mx.log_softmax(valid_logits, axis=-1)
        
        # Gather log probabilities for targets
        batch_size = valid_targets.shape[0]
        target_log_probs = log_probs[mx.arange(batch_size), valid_targets]
        
        # Return mean negative log likelihood
        return -mx.mean(target_log_probs)


class DataLoader_MLX:
    """Simple data loader for MLX"""
    
    def __init__(self, 
                 data_list: List[Any],
                 batch_size: int = 32,
                 shuffle: bool = True):
        self.data_list = data_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(data_list)))
    
    def __iter__(self) -> Iterator[List[Any]]:
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch = [self.data_list[idx] for idx in batch_indices]
            yield batch
    
    def __len__(self) -> int:
        return (len(self.data_list) + self.batch_size - 1) // self.batch_size


class GNNTrainer_MLX:
    """Trainer for Graph Neural Network models in MLX"""
    
    def __init__(self,
                 model: ParentPredictionGNN_MLX,
                 config: Dict[str, Any]):
        self.model = model
        self.config = config
        
        # Optimizer
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.optimizer = optim.Adam(learning_rate=self.learning_rate)
        
        # Loss function
        self.criterion = ParentPredictionLoss_MLX(
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 0.5),
            gamma=config.get('gamma', 0.1)
        )
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints_mlx'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Metrics tracking
        self.best_val_loss = float('inf')
        self.training_history = []
    
    def train_step(self, batch: GraphData) -> Dict[str, float]:
        """Single training step"""
        def loss_fn(model):
            predictions = model(batch)
            losses = self.criterion(predictions, batch.y, batch.train_mask, batch)
            return losses['total_loss'], losses
        
        # Compute loss and gradients
        (loss_value, losses), grads = mx.value_and_grad(loss_fn, has_aux=True)(self.model)
        
        # Update parameters
        self.optimizer.update(self.model, grads)
        
        # Convert to Python floats for logging
        loss_dict = {k: float(v) for k, v in losses.items()}
        
        return loss_dict
    
    def validate(self, val_data: List[GraphData]) -> Dict[str, float]:
        """Validate the model"""
        total_losses = {
            'total_loss': 0.0,
            'parent_loss': 0.0,
            'generation_loss': 0.0,
            'consistency_loss': 0.0
        }
        
        all_predictions = []
        all_targets = []
        
        for batch in val_data:
            # Forward pass
            predictions = self.model(batch)
            
            # Compute losses
            losses = self.criterion(predictions, batch.y, batch.val_mask, batch)
            
            # Accumulate losses
            for key in total_losses:
                if key in losses:
                    total_losses[key] += float(losses[key])
            
            # Store predictions
            father_preds = mx.argmax(predictions['father_scores'], axis=1)
            mother_preds = mx.argmax(predictions['mother_scores'], axis=1)
            preds = mx.stack([father_preds, mother_preds], axis=1)
            
            all_predictions.append(preds[batch.val_mask])
            all_targets.append(batch.y[batch.val_mask])
        
        # Average losses
        num_batches = len(val_data)
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        # Compute metrics
        if all_predictions:
            all_predictions = mx.concatenate(all_predictions, axis=0)
            all_targets = mx.concatenate(all_targets, axis=0)
            metrics = compute_metrics_mlx(all_predictions, all_targets)
            avg_losses.update({f'val_{k}': v for k, v in metrics.items()})
        
        return avg_losses
    
    def train(self,
              train_data: List[GraphData],
              val_data: List[GraphData],
              num_epochs: int = 100):
        """Full training loop"""
        
        train_loader = DataLoader_MLX(train_data, self.config.get('batch_size', 32))
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            epoch_losses = {
                'total_loss': 0.0,
                'parent_loss': 0.0,
                'generation_loss': 0.0,
                'consistency_loss': 0.0
            }
            
            for batch_list in tqdm(train_loader, desc="Training"):
                # For simplicity, process one graph at a time
                # In production, you'd want to batch properly
                for graph in batch_list:
                    losses = self.train_step(graph)
                    for key in epoch_losses:
                        if key in losses:
                            epoch_losses[key] += losses[key]
            
            # Average training losses
            num_train_batches = len(train_data)
            avg_train_losses = {
                f'train_{k}': v / num_train_batches 
                for k, v in epoch_losses.items()
            }
            
            # Validation
            val_metrics = self.validate(val_data)
            
            # Log metrics
            all_metrics = {**avg_train_losses, **val_metrics}
            self._log_metrics(all_metrics, epoch)
            
            # Save checkpoint if improved
            if val_metrics.get('total_loss', float('inf')) < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self._save_checkpoint(epoch, val_metrics)
            
            # Early stopping check
            if self._should_stop_early(epoch):
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    def _log_metrics(self, metrics: Dict[str, float], epoch: int):
        """Log metrics"""
        self.training_history.append({
            'epoch': epoch,
            **metrics
        })
        
        # Console logging
        log_str = f"Epoch {epoch+1}: "
        log_str += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(log_str)
        
        # Save history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.parameters(),
            'optimizer_state': self.optimizer.state,
            'metrics': metrics,
            'config': self.config
        }
        
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}_loss_{metrics["total_loss"]:.4f}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Also save as best
        best_path = self.checkpoint_dir / 'best_model.pkl'
        with open(best_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"Saved checkpoint: {path}")
    
    def _should_stop_early(self, epoch: int) -> bool:
        """Check if should stop early"""
        if len(self.training_history) < self.config.get('early_stopping_patience', 20):
            return False
        
        recent_losses = [
            h.get('val_total_loss', h.get('total_loss', float('inf')))
            for h in self.training_history[-self.config.get('early_stopping_patience', 20):]
        ]
        
        # Check if loss hasn't improved
        return all(loss >= self.best_val_loss for loss in recent_losses)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore model parameters
        self.model.update(checkpoint['model_state'])
        
        # Restore optimizer state
        self.optimizer = optim.Adam(learning_rate=self.learning_rate)
        self.optimizer.state = checkpoint['optimizer_state']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


class TransformerTrainer_MLX:
    """Trainer for Transformer models in MLX"""
    
    def __init__(self,
                 model: ParentPredictionTransformer_MLX,
                 config: Dict[str, Any]):
        self.model = model
        self.config = config
        
        # Optimizer with warmup
        self.base_lr = config.get('learning_rate', 1e-4)
        self.warmup_steps = config.get('warmup_steps', 1000)
        self.current_step = 0
        
        self.optimizer = optim.AdamW(
            learning_rate=self._get_lr(),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Loss function
        self.criterion = ParentPredictionLoss_MLX(
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 0.5),
            gamma=config.get('gamma', 0.1)
        )
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints_mlx'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.training_history = []
    
    def _get_lr(self) -> float:
        """Get learning rate with warmup"""
        if self.current_step < self.warmup_steps:
            return self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay after warmup
            progress = (self.current_step - self.warmup_steps) / \
                      (self.config.get('total_steps', 10000) - self.warmup_steps)
            return self.base_lr * (0.5 * (1 + np.cos(np.pi * progress)))
    
    def train_step(self, batch: Dict[str, mx.array]) -> Dict[str, float]:
        """Single training step for transformer"""
        self.current_step += 1
        
        # Update learning rate
        new_lr = self._get_lr()
        self.optimizer.learning_rate = new_lr
        
        def loss_fn(model):
            predictions = model(
                batch['features'],
                mask=batch.get('mask'),
                generation_info=batch.get('generation_info'),
                gender_info=batch.get('gender_info')
            )
            
            losses = self.criterion(
                predictions,
                batch['targets'],
                batch['train_mask'],
                None  # No graph data for transformer
            )
            
            return losses['total_loss'], losses
        
        # Compute loss and gradients
        (loss_value, losses), grads = mx.value_and_grad(loss_fn, has_aux=True)(self.model)
        
        # Gradient clipping
        grads = self._clip_gradients(grads, self.config.get('grad_clip', 1.0))
        
        # Update parameters
        self.optimizer.update(self.model, grads)
        
        # Convert to Python floats
        loss_dict = {k: float(v) for k, v in losses.items()}
        loss_dict['learning_rate'] = new_lr
        
        return loss_dict
    
    def _clip_gradients(self, grads: Dict, max_norm: float) -> Dict:
        """Clip gradients by norm"""
        # Compute total norm
        total_norm = 0.0
        for g in grads.values():
            if g is not None:
                total_norm += mx.sum(g ** 2)
        total_norm = mx.sqrt(total_norm)
        
        # Clip if needed
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            grads = {k: g * clip_coef if g is not None else None 
                    for k, g in grads.items()}
        
        return grads
    
    def train(self,
              train_data: List[Dict[str, mx.array]],
              val_data: List[Dict[str, mx.array]],
              num_epochs: int = 100):
        """Full training loop for transformer"""
        
        train_loader = DataLoader_MLX(train_data, self.config.get('batch_size', 16))
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            epoch_losses = {}
            
            for batch_list in tqdm(train_loader, desc="Training Transformer"):
                for batch in batch_list:
                    losses = self.train_step(batch)
                    
                    # Accumulate losses
                    for k, v in losses.items():
                        if k not in epoch_losses:
                            epoch_losses[k] = []
                        epoch_losses[k].append(v)
            
            # Average losses
            avg_train_losses = {
                f'train_{k}': np.mean(v) 
                for k, v in epoch_losses.items()
            }
            
            # Validation
            val_metrics = self.validate(val_data)
            
            # Log metrics
            all_metrics = {**avg_train_losses, **val_metrics}
            self._log_metrics(all_metrics, epoch)
            
            # Save checkpoint if improved
            if val_metrics.get('total_loss', float('inf')) < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self._save_checkpoint(epoch, val_metrics)
    
    def validate(self, val_data: List[Dict[str, mx.array]]) -> Dict[str, float]:
        """Validate transformer model"""
        total_losses = {}
        
        for batch in val_data:
            predictions = self.model(
                batch['features'],
                mask=batch.get('mask'),
                generation_info=batch.get('generation_info'),
                gender_info=batch.get('gender_info')
            )
            
            losses = self.criterion(
                predictions,
                batch['targets'],
                batch['val_mask'],
                None
            )
            
            # Accumulate
            for k, v in losses.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += float(v)
        
        # Average
        avg_losses = {
            f'val_{k}': v / len(val_data) 
            for k, v in total_losses.items()
        }
        
        return avg_losses
    
    def _log_metrics(self, metrics: Dict[str, float], epoch: int):
        """Log metrics"""
        self.training_history.append({
            'epoch': epoch,
            'step': self.current_step,
            **metrics
        })
        
        # Console logging
        important_metrics = ['train_total_loss', 'val_total_loss', 'learning_rate']
        log_parts = []
        for metric in important_metrics:
            if metric in metrics:
                log_parts.append(f"{metric}: {metrics[metric]:.4f}")
        
        print(f"Epoch {epoch+1}: " + " | ".join(log_parts))
        
        # Save history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': self.current_step,
            'model_state': self.model.parameters(),
            'optimizer_state': {
                'step': self.current_step,
                'lr': self.optimizer.learning_rate
            },
            'metrics': metrics,
            'config': self.config
        }
        
        path = self.checkpoint_dir / f'transformer_epoch_{epoch}_loss_{metrics["val_total_loss"]:.4f}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Also save as best
        best_path = self.checkpoint_dir / 'best_transformer.pkl'
        with open(best_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"Saved checkpoint: {path}")


class IncrementalTrainer_MLX:
    """Incremental learning trainer for MLX"""
    
    def __init__(self,
                 model,
                 base_trainer,
                 config: Dict[str, Any]):
        self.model = model
        self.base_trainer = base_trainer
        self.config = config
        
        # Experience replay buffer
        self.replay_buffer = []
        self.buffer_size = config.get('replay_buffer_size', 1000)
        
        # For preventing catastrophic forgetting
        self.old_model_params = None
        self.distillation_weight = config.get('distillation_weight', 0.1)
    
    def update_with_new_data(self,
                            new_data: List[Any],
                            old_data_sample: Optional[List[Any]] = None):
        """Update model with new GEDCOM data"""
        
        # Store old model parameters for distillation
        if self.distillation_weight > 0:
            self.old_model_params = self._copy_model_params()
        
        # Combine old and new data
        if old_data_sample:
            combined_data = old_data_sample + new_data
        else:
            combined_data = new_data
        
        # Create modified loss function with distillation
        original_criterion = self.base_trainer.criterion
        
        def distillation_loss_wrapper(predictions, targets, mask, data=None):
            # Original loss
            losses = original_criterion(predictions, targets, mask, data)
            
            # Knowledge distillation loss
            if self.old_model_params is not None:
                # Compute old model predictions
                old_predictions = self._get_old_model_predictions(data)
                
                # KL divergence between old and new predictions
                kl_loss = self._kl_divergence(
                    predictions['father_scores'],
                    old_predictions['father_scores']
                ) + self._kl_divergence(
                    predictions['mother_scores'],
                    old_predictions['mother_scores']
                )
                
                losses['distillation_loss'] = self.distillation_weight * kl_loss
                losses['total_loss'] = losses['total_loss'] + losses['distillation_loss']
            
            return losses
        
        # Temporarily replace criterion
        self.base_trainer.criterion = distillation_loss_wrapper
        
        # Train on combined data
        self.base_trainer.train_step(combined_data)
        
        # Restore original criterion
        self.base_trainer.criterion = original_criterion
        
        # Update replay buffer
        self._update_replay_buffer(new_data)
    
    def _copy_model_params(self) -> Dict:
        """Copy model parameters"""
        return {k: v.copy() for k, v in self.model.parameters().items()}
    
    def _get_old_model_predictions(self, data) -> Dict[str, mx.array]:
        """Get predictions from old model"""
        # Temporarily set old parameters
        current_params = self.model.parameters()
        self.model.update(self.old_model_params)
        
        # Get predictions
        predictions = self.model(data)
        
        # Restore current parameters
        self.model.update(current_params)
        
        return predictions
    
    def _kl_divergence(self, p_logits: mx.array, q_logits: mx.array) -> mx.array:
        """Compute KL divergence between two distributions"""
        p = mx.softmax(p_logits, axis=-1)
        q = mx.softmax(q_logits, axis=-1)
        
        # KL(P||Q) = sum(P * log(P/Q))
        return mx.sum(p * (mx.log(p + 1e-8) - mx.log(q + 1e-8)), axis=-1).mean()
    
    def _update_replay_buffer(self, new_data: List[Any]):
        """Update replay buffer with new samples"""
        for item in new_data:
            if len(self.replay_buffer) >= self.buffer_size:
                self.replay_buffer.pop(0)
            self.replay_buffer.append(item)
    
    def get_replay_sample(self, sample_size: int) -> List[Any]:
        """Get a sample from replay buffer"""
        if len(self.replay_buffer) == 0:
            return []
        
        sample_size = min(sample_size, len(self.replay_buffer))
        indices = np.random.choice(len(self.replay_buffer), sample_size, replace=False)
        
        return [self.replay_buffer[i] for i in indices]