"""
Metrics and Evaluation Utilities for ʻŌhanaGPT
"""

import torch
import mlx.core as mx
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def compute_metrics(predictions: torch.Tensor, 
                   targets: torch.Tensor,
                   top_k: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Compute various metrics for parent prediction
    
    Args:
        predictions: Predicted parent indices [N, 2] (father, mother)
        targets: True parent indices [N, 2]
        top_k: List of k values for top-k accuracy
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Filter out invalid targets
    valid_mask = (targets >= 0).all(dim=1)
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    
    if len(predictions) == 0:
        return {f'top_{k}_accuracy': 0.0 for k in top_k}
    
    # Exact match accuracy
    exact_match = (predictions == targets).all(dim=1).float().mean()
    metrics['exact_match_accuracy'] = exact_match.item()
    
    # Father and mother accuracy separately
    father_acc = (predictions[:, 0] == targets[:, 0]).float().mean()
    mother_acc = (predictions[:, 1] == targets[:, 1]).float().mean()
    metrics['father_accuracy'] = father_acc.item()
    metrics['mother_accuracy'] = mother_acc.item()
    
    # For top-k metrics, we need the full score matrices
    # This is a simplified version - in practice, you'd pass the score matrices
    for k in top_k:
        metrics[f'top_{k}_accuracy'] = exact_match.item()  # Placeholder
    
    return metrics


def compute_metrics_mlx(predictions: mx.array, 
                       targets: mx.array,
                       top_k: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Compute metrics for MLX arrays
    """
    # Convert to numpy for easier computation
    predictions_np = np.array(predictions)
    targets_np = np.array(targets)
    
    metrics = {}
    
    # Filter out invalid targets
    valid_mask = (targets_np >= 0).all(axis=1)
    predictions_np = predictions_np[valid_mask]
    targets_np = targets_np[valid_mask]
    
    if len(predictions_np) == 0:
        return {f'top_{k}_accuracy': 0.0 for k in top_k}
    
    # Exact match accuracy
    exact_match = (predictions_np == targets_np).all(axis=1).mean()
    metrics['exact_match_accuracy'] = float(exact_match)
    
    # Father and mother accuracy separately
    father_acc = (predictions_np[:, 0] == targets_np[:, 0]).mean()
    mother_acc = (predictions_np[:, 1] == targets_np[:, 1]).mean()
    metrics['father_accuracy'] = float(father_acc)
    metrics['mother_accuracy'] = float(mother_acc)
    
    return metrics


def evaluate_predictions(model, 
                        test_data,
                        device: str = 'cuda',
                        top_k: int = 10) -> Dict[str, any]:
    """
    Comprehensive evaluation of model predictions
    
    Returns detailed evaluation results including:
    - Accuracy metrics
    - Confidence distributions
    - Error analysis
    """
    model.eval()
    
    all_results = {
        'predictions': [],
        'targets': [],
        'confidence_scores': [],
        'metadata': []
    }
    
    with torch.no_grad():
        for batch in test_data:
            batch = batch.to(device)
            
            # Get predictions
            outputs = model.predict_parents(batch, top_k=top_k)
            
            # Store results
            all_results['predictions'].append(outputs['father_indices'][:, 0])  # Top-1
            all_results['predictions'].append(outputs['mother_indices'][:, 0])
            all_results['targets'].append(batch.y)
            all_results['confidence_scores'].append(outputs['father_scores'][:, 0])
            all_results['confidence_scores'].append(outputs['mother_scores'][:, 0])
            
            # Store metadata for error analysis
            if hasattr(batch, 'generation'):
                all_results['metadata'].append({
                    'generation': batch.generation,
                    'has_dates': hasattr(batch, 'birth_years')
                })
    
    # Compute aggregate metrics
    predictions = torch.cat([p.unsqueeze(1) if p.dim() == 1 else p 
                           for p in all_results['predictions']], dim=0)
    targets = torch.cat(all_results['targets'], dim=0)
    confidence = torch.cat(all_results['confidence_scores'], dim=0)
    
    metrics = compute_metrics(predictions, targets)
    
    # Confidence analysis
    metrics['mean_confidence'] = confidence.mean().item()
    metrics['confidence_std'] = confidence.std().item()
    
    # Error analysis
    errors = analyze_errors(predictions, targets, all_results['metadata'])
    metrics['error_analysis'] = errors
    
    return metrics


def analyze_errors(predictions: torch.Tensor,
                  targets: torch.Tensor,
                  metadata: List[Dict]) -> Dict[str, any]:
    """
    Analyze prediction errors to identify patterns
    """
    errors = {
        'by_generation': defaultdict(list),
        'by_data_completeness': defaultdict(list),
        'common_mistakes': []
    }
    
    # Identify errors
    is_error = (predictions != targets).any(dim=1)
    error_indices = torch.where(is_error)[0]
    
    # Analyze by generation if metadata available
    if metadata and 'generation' in metadata[0]:
        generations = torch.cat([m['generation'] for m in metadata])
        
        for idx in error_indices:
            gen = generations[idx].item()
            errors['by_generation'][int(gen)].append(idx.item())
    
    # Analyze by data completeness
    if metadata and 'has_dates' in metadata[0]:
        for idx in error_indices:
            has_dates = metadata[idx // len(metadata[0])]['has_dates']
            errors['by_data_completeness']['with_dates' if has_dates else 'without_dates'].append(idx.item())
    
    # Find common error patterns
    # This is simplified - in practice, you'd do more sophisticated analysis
    
    return errors


def plot_training_history(history: List[Dict], save_path: Optional[str] = None):
    """
    Plot training history including losses and metrics
    """
    epochs = [h['epoch'] for h in history]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total loss
    train_loss = [h.get('train_loss', h.get('train_total_loss', 0)) for h in history]
    val_loss = [h.get('val_loss', h.get('val_total_loss', 0)) for h in history]
    
    axes[0, 0].plot(epochs, train_loss, label='Train')
    axes[0, 0].plot(epochs, val_loss, label='Validation')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Parent accuracy
    if 'val_father_accuracy' in history[0]:
        father_acc = [h.get('val_father_accuracy', 0) for h in history]
        mother_acc = [h.get('val_mother_accuracy', 0) for h in history]
        
        axes[0, 1].plot(epochs, father_acc, label='Father')
        axes[0, 1].plot(epochs, mother_acc, label='Mother')
        axes[0, 1].set_title('Parent Prediction Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
    
    # Learning rate
    if 'learning_rate' in history[0]:
        lr = [h.get('learning_rate', 0) for h in history]
        axes[1, 0].plot(epochs, lr)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].set_yscale('log')
    
    # Component losses
    if 'train_generation_loss' in history[0]:
        gen_loss = [h.get('train_generation_loss', 0) for h in history]
        cons_loss = [h.get('train_consistency_loss', 0) for h in history]
        
        axes[1, 1].plot(epochs, gen_loss, label='Generation')
        axes[1, 1].plot(epochs, cons_loss, label='Consistency')
        axes[1, 1].set_title('Component Losses')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    return fig


def plot_prediction_confidence_distribution(confidence_scores: np.ndarray,
                                          is_correct: np.ndarray,
                                          save_path: Optional[str] = None):
    """
    Plot distribution of confidence scores for correct vs incorrect predictions
    """
    plt.figure(figsize=(10, 6))
    
    # Separate correct and incorrect predictions
    correct_conf = confidence_scores[is_correct]
    incorrect_conf = confidence_scores[~is_correct]
    
    # Plot histograms
    plt.hist(correct_conf, bins=50, alpha=0.6, label='Correct', density=True)
    plt.hist(incorrect_conf, bins=50, alpha=0.6, label='Incorrect', density=True)
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    
    # Add vertical lines for means
    plt.axvline(correct_conf.mean(), color='green', linestyle='--', 
                label=f'Correct Mean: {correct_conf.mean():.3f}')
    plt.axvline(incorrect_conf.mean(), color='red', linestyle='--',
                label=f'Incorrect Mean: {incorrect_conf.mean():.3f}')
    
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def compute_relationship_metrics(predictions: Dict[str, torch.Tensor],
                               family_data: Dict,
                               edge_index: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics specific to family relationships
    """
    metrics = {}
    
    # Check if predicted parents are actually spouses
    father_preds = predictions['father_indices'][:, 0]
    mother_preds = predictions['mother_indices'][:, 0]
    
    # Build spouse relationships from edge data
    spouse_pairs = set()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        # Assuming edge_attr indicates relationship type
        # This is simplified - you'd need actual relationship type info
        spouse_pairs.add((min(src, dst), max(src, dst)))
    
    # Check how many predicted parent pairs are spouses
    valid_spouse_count = 0
    for f, m in zip(father_preds, mother_preds):
        if (min(f.item(), m.item()), max(f.item(), m.item())) in spouse_pairs:
            valid_spouse_count += 1
    
    metrics['parent_spouse_consistency'] = valid_spouse_count / len(father_preds)
    
    # Check generation consistency
    # This requires generation information in the data
    
    return metrics


def create_evaluation_report(model,
                           test_data,
                           output_path: str,
                           device: str = 'cuda'):
    """
    Create a comprehensive evaluation report
    """
    print("Generating evaluation report...")
    
    # Get predictions and metrics
    results = evaluate_predictions(model, test_data, device)
    
    # Create report
    report = []
    report.append("# ʻŌhanaGPT Evaluation Report")
    report.append(f"\nGenerated on: {np.datetime64('now')}")
    report.append("\n## Overall Metrics")
    
    for metric, value in results.items():
        if metric != 'error_analysis':
            report.append(f"- {metric}: {value:.4f}")
    
    report.append("\n## Error Analysis")
    error_analysis = results['error_analysis']
    
    report.append("\n### Errors by Generation")
    for gen, errors in error_analysis['by_generation'].items():
        report.append(f"- Generation {gen}: {len(errors)} errors")
    
    report.append("\n### Errors by Data Completeness")
    for category, errors in error_analysis['by_data_completeness'].items():
        report.append(f"- {category}: {len(errors)} errors")
    
    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report saved to {output_path}")
    
    return results