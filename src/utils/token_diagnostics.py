"""
Token Usage and FSQ Geometric Distance Diagnostics

This module provides diagnostic functions to analyze:
1. Token usage frequency (GT vs predictions) - detect mode collapse
2. FSQ geometric distances for mispredicted tokens - measure prediction quality
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from collections import Counter


@torch.no_grad()
def analyze_token_usage(
    pred_indices: Dict[str, torch.Tensor],
    gt_indices: Dict[str, torch.Tensor],
    null_idx: int,
    vocab_size: int,
    scales: Optional[list] = None
) -> Dict[str, Dict]:
    """
    Analyze token usage frequency for GT vs predictions.
    
    Detects if predictions are falling back to specific tokens (mode collapse).
    
    Args:
        pred_indices: Dict mapping scale names to predicted indices [B, S, D, H, W]
        gt_indices: Dict mapping scale names to GT indices [B, S, D, H, W]
        null_idx: Index of the null/zero token
        vocab_size: Total vocabulary size
        scales: List of scales to analyze (default: all keys in pred_indices)
    
    Returns:
        Dict mapping scale names to statistics:
        - token_usage_gt: Counter of GT token frequencies (excluding null)
        - token_usage_pred: Counter of predicted token frequencies (excluding null)
        - unique_tokens_gt: Number of unique tokens in GT
        - unique_tokens_pred: Number of unique tokens in predictions
        - top_k_tokens_gt: Top K most frequent tokens in GT
        - top_k_tokens_pred: Top K most frequent tokens in predictions
        - token_overlap: Set intersection of tokens used in GT and predictions
        - token_coverage: Fraction of GT tokens that are also predicted
    """
    if scales is None:
        scales = list(pred_indices.keys())
    
    results = {}
    
    for scale in scales:
        if scale not in pred_indices or scale not in gt_indices:
            continue
        
        pred = pred_indices[scale]  # [B, S, D, H, W]
        gt = gt_indices[scale]      # [B, S, D, H, W]
        
        # Flatten to 1D for counting
        pred_flat = pred.flatten().cpu().numpy()
        gt_flat = gt.flatten().cpu().numpy()
        
        # Filter out null tokens
        pred_active = pred_flat[pred_flat != null_idx]
        gt_active = gt_flat[gt_flat != null_idx]
        
        # Count token frequencies
        token_usage_gt = Counter(gt_active.astype(int).tolist())
        token_usage_pred = Counter(pred_active.astype(int).tolist())
        
        # Statistics
        unique_tokens_gt = len(token_usage_gt)
        unique_tokens_pred = len(token_usage_pred)
        
        # Top K tokens (K=10)
        top_k = 10
        top_k_tokens_gt = dict(token_usage_gt.most_common(top_k))
        top_k_tokens_pred = dict(token_usage_pred.most_common(top_k))
        
        # Token overlap
        tokens_gt_set = set(token_usage_gt.keys())
        tokens_pred_set = set(token_usage_pred.keys())
        token_overlap = tokens_gt_set & tokens_pred_set
        token_coverage = len(token_overlap) / (len(tokens_gt_set) + 1e-8)
        
        # Calculate entropy (measure of diversity)
        def calculate_entropy(counter, vocab_size):
            """Calculate Shannon entropy in bits."""
            total = sum(counter.values())
            if total == 0:
                return 0.0
            probs = [count / total for count in counter.values()]
            return -sum(p * np.log2(p) for p in probs if p > 0)
        
        entropy_gt = calculate_entropy(token_usage_gt, vocab_size)
        entropy_pred = calculate_entropy(token_usage_pred, vocab_size)
        entropy_ratio = entropy_pred / (entropy_gt + 1e-8)
        
        results[scale] = {
            'token_usage_gt': token_usage_gt,
            'token_usage_pred': token_usage_pred,
            'unique_tokens_gt': unique_tokens_gt,
            'unique_tokens_pred': unique_tokens_pred,
            'top_k_tokens_gt': top_k_tokens_gt,
            'top_k_tokens_pred': top_k_tokens_pred,
            'token_overlap': token_overlap,
            'token_coverage': token_coverage,
            'entropy_gt': entropy_gt,
            'entropy_pred': entropy_pred,
            'entropy_ratio': entropy_ratio,
            'num_active_pred': len(pred_active),
            'num_active_gt': len(gt_active),
        }
    
    return results


@torch.no_grad()
def calculate_fsq_geometric_distances(
    pred_indices: Dict[str, torch.Tensor],
    gt_indices: Dict[str, torch.Tensor],
    fsq,
    null_idx: int,
    scales: Optional[list] = None,
    device: str = 'cuda'
) -> Dict[str, Dict]:
    """
    Calculate FSQ geometric distances for mispredicted tokens.
    
    Since FSQ is a geometric lattice, close tokens are better than far away tokens.
    This measures prediction quality beyond simple accuracy.
    
    Args:
        pred_indices: Dict mapping scale names to predicted indices [B, S, D, H, W]
        gt_indices: Dict mapping scale names to GT indices [B, S, D, H, W]
        fsq: FSQ module instance (from tokenizer.fsq)
        null_idx: Index of the null/zero token
        scales: List of scales to analyze (default: all keys in pred_indices)
        device: Device to run computations on
    
    Returns:
        Dict mapping scale names to statistics:
        - exact_match_rate: Fraction of predictions that exactly match GT
        - mean_distance: Mean Euclidean distance for all active predictions
        - mean_distance_mispredicted: Mean distance for mispredicted tokens only
        - median_distance: Median distance for all active predictions
        - p95_distance: 95th percentile distance
        - p99_distance: 99th percentile distance
        - distance_histogram: Histogram of distances (for visualization)
    """
    if scales is None:
        scales = list(pred_indices.keys())
    
    # Pre-compute all codebook vectors [vocab_size, dim]
    vocab_size = fsq.vocab_size
    all_indices = torch.arange(vocab_size, device=device).view(vocab_size, 1, 1)
    codebook = fsq.indices_to_codes(all_indices)  # [vocab_size, dim, 1, 1]
    codebook = codebook.view(vocab_size, -1)  # [vocab_size, dim]
    
    results = {}
    
    for scale in scales:
        if scale not in pred_indices or scale not in gt_indices:
            continue
        
        pred = pred_indices[scale].to(device)  # [B, S, D, H, W]
        gt = gt_indices[scale].to(device)      # [B, S, D, H, W]
        
        # Ensure pred and gt have the same shape (handle mismatches in any dimension)
        if pred.shape != gt.shape:
            # Take minimum dimensions across all axes
            if pred.ndim == gt.ndim:
                # Same number of dimensions, align each dimension
                slices = tuple(slice(0, min(pred.shape[i], gt.shape[i])) for i in range(pred.ndim))
                pred = pred[slices]
                gt = gt[slices]
            else:
                # Different number of dimensions - this shouldn't happen, but handle gracefully
                # Flatten both and take minimum length
                pred_flat_temp = pred.flatten()
                gt_flat_temp = gt.flatten()
                min_len = min(len(pred_flat_temp), len(gt_flat_temp))
                pred_flat = pred_flat_temp[:min_len]
                gt_flat = gt_flat_temp[:min_len]
                
                # Filter to active positions
                active_mask = (gt_flat != null_idx)
                pred_active = pred_flat[active_mask]
                gt_active = gt_flat[active_mask]
                
                if len(pred_active) == 0:
                    results[scale] = {
                        'exact_match_rate': 0.0,
                        'mean_distance': float('inf'),
                        'mean_distance_mispredicted': float('inf'),
                        'median_distance': float('inf'),
                        'p95_distance': float('inf'),
                        'p99_distance': float('inf'),
                        'num_active': 0,
                    }
                    continue
                
                # Get codebook vectors
                vocab_size = fsq.vocab_size
                all_indices = torch.arange(vocab_size, device=device).view(vocab_size, 1, 1)
                codebook = fsq.indices_to_codes(all_indices).view(vocab_size, -1)
                
                pred_active_clamped = torch.clamp(pred_active, 0, vocab_size - 1)
                gt_active_clamped = torch.clamp(gt_active, 0, vocab_size - 1)
                
                pred_codes = codebook[pred_active_clamped]
                gt_codes = codebook[gt_active_clamped]
                
                distances = torch.norm(pred_codes - gt_codes, dim=1)
                exact_matches = (pred_active == gt_active).float()
                
                distances_np = distances.cpu().numpy()
                mispredicted_mask = (pred_active != gt_active)
                
                results[scale] = {
                    'exact_match_rate': exact_matches.mean().item(),
                    'mean_distance': distances.mean().item(),
                    'mean_distance_mispredicted': distances[mispredicted_mask].mean().item() if mispredicted_mask.any() else 0.0,
                    'median_distance': float(np.median(distances_np)),
                    'p95_distance': float(np.percentile(distances_np, 95)),
                    'p99_distance': float(np.percentile(distances_np, 99)),
                    'num_active': len(pred_active),
                }
                continue
        
        # Flatten to 1D (now pred and gt have same shape)
        pred_flat = pred.flatten()  # [N]
        gt_flat = gt.flatten()      # [N]
        
        # Filter to active (non-null) positions where GT is active
        active_mask = (gt_flat != null_idx)
        pred_active = pred_flat[active_mask]
        gt_active = gt_flat[active_mask]
        
        if len(pred_active) == 0:
            results[scale] = {
                'exact_match_rate': 0.0,
                'mean_distance': float('inf'),
                'mean_distance_mispredicted': float('inf'),
                'median_distance': float('inf'),
                'p95_distance': float('inf'),
                'p99_distance': float('inf'),
                'num_active': 0,
            }
            continue
        
        # Get codebook vectors for GT and predictions
        # Clamp indices to valid range [0, vocab_size-1]
        pred_active_clamped = torch.clamp(pred_active, 0, vocab_size - 1)
        gt_active_clamped = torch.clamp(gt_active, 0, vocab_size - 1)
        
        pred_codes = codebook[pred_active_clamped]  # [N, dim]
        gt_codes = codebook[gt_active_clamped]      # [N, dim]
        
        # Calculate Euclidean distances
        distances = torch.norm(pred_codes - gt_codes, dim=1)  # [N]
        
        # Exact match rate
        exact_matches = (pred_active == gt_active).float()
        exact_match_rate = exact_matches.mean().item()
        
        # Distance statistics
        distances_np = distances.cpu().numpy()
        mean_distance = distances.mean().item()
        median_distance = np.median(distances_np)
        p95_distance = np.percentile(distances_np, 95)
        p99_distance = np.percentile(distances_np, 99)
        
        # Distance for mispredicted tokens only
        mispredicted_mask = (pred_active != gt_active)
        if mispredicted_mask.sum() > 0:
            mispredicted_distances = distances[mispredicted_mask]
            mean_distance_mispredicted = mispredicted_distances.mean().item()
        else:
            mean_distance_mispredicted = 0.0
        
        # Histogram for visualization (bins: 0 to max_distance)
        max_dist = distances.max().item()
        num_bins = 50
        histogram, bin_edges = np.histogram(distances_np, bins=num_bins, range=(0, max_dist))
        
        results[scale] = {
            'exact_match_rate': exact_match_rate,
            'mean_distance': mean_distance,
            'mean_distance_mispredicted': mean_distance_mispredicted,
            'median_distance': median_distance,
            'p95_distance': p95_distance,
            'p99_distance': p99_distance,
            'max_distance': max_dist,
            'distance_histogram': (histogram, bin_edges),
            'num_active': len(pred_active),
        }
    
    return results


@torch.no_grad()
def comprehensive_token_diagnostics(
    model,
    pred_indices: Dict[str, torch.Tensor],
    gt_indices: Dict[str, torch.Tensor],
    scales: Optional[list] = None,
    device: str = 'cuda'
) -> Dict[str, Dict]:
    """
    Comprehensive token diagnostics combining usage and geometric distance analysis.
    
    Args:
        model: Model instance (needs model.tokenizer.fsq and model.tokenizer.null_idx)
        pred_indices: Dict mapping scale names to predicted indices [B, S, D, H, W]
        gt_indices: Dict mapping scale names to GT indices [B, S, D, H, W]
        scales: List of scales to analyze (default: all keys in pred_indices)
        device: Device to run computations on
    
    Returns:
        Dict with two keys:
        - 'token_usage': Results from analyze_token_usage
        - 'geometric_distances': Results from calculate_fsq_geometric_distances
    """
    # Get FSQ and null_idx from model
    fsq = model.tokenizer.fsq
    null_idx = model.tokenizer.null_idx
    vocab_size = fsq.vocab_size
    
    # Token usage analysis
    usage_results = analyze_token_usage(
        pred_indices=pred_indices,
        gt_indices=gt_indices,
        null_idx=null_idx,
        vocab_size=vocab_size,
        scales=scales
    )
    
    # Geometric distance analysis
    distance_results = calculate_fsq_geometric_distances(
        pred_indices=pred_indices,
        gt_indices=gt_indices,
        fsq=fsq,
        null_idx=null_idx,
        scales=scales,
        device=device
    )
    
    return {
        'token_usage': usage_results,
        'geometric_distances': distance_results,
    }


def print_token_diagnostics(diagnostics: Dict[str, Dict], scales: Optional[list] = None):
    """
    Pretty print token diagnostics results.
    
    Args:
        diagnostics: Output from comprehensive_token_diagnostics
        scales: List of scales to print (default: all)
    """
    usage_results = diagnostics['token_usage']
    distance_results = diagnostics['geometric_distances']
    
    if scales is None:
        scales = list(usage_results.keys())
    
    print("=" * 80)
    print("TOKEN USAGE & FSQ GEOMETRIC DISTANCE DIAGNOSTICS")
    print("=" * 80)
    
    for scale in scales:
        if scale not in usage_results or scale not in distance_results:
            continue
        
        usage = usage_results[scale]
        distances = distance_results[scale]
        
        print(f"\n{scale.upper()}:")
        print("-" * 80)
        
        # Token Usage
        print("\n📊 Token Usage:")
        print(f"  Unique tokens (GT):     {usage['unique_tokens_gt']:5d} / {usage.get('vocab_size', '?')}")
        print(f"  Unique tokens (Pred):   {usage['unique_tokens_pred']:5d} / {usage.get('vocab_size', '?')}")
        print(f"  Token coverage:         {usage['token_coverage']:.2%} (GT tokens also in predictions)")
        print(f"  Entropy (GT):           {usage['entropy_gt']:.3f} bits")
        print(f"  Entropy (Pred):         {usage['entropy_pred']:.3f} bits")
        print(f"  Entropy ratio:          {usage['entropy_ratio']:.3f} (1.0 = perfect match)")
        
        # Top tokens
        print(f"\n  Top 5 tokens (GT):     {list(usage['top_k_tokens_gt'].keys())[:5]}")
        print(f"  Top 5 tokens (Pred):   {list(usage['top_k_tokens_pred'].keys())[:5]}")
        
        # Geometric Distances
        print("\n📏 FSQ Geometric Distances:")
        print(f"  Exact match rate:       {distances['exact_match_rate']:.2%}")
        print(f"  Mean distance:          {distances['mean_distance']:.4f}")
        print(f"  Mean (mispredicted):    {distances['mean_distance_mispredicted']:.4f}")
        print(f"  Median distance:        {distances['median_distance']:.4f}")
        print(f"  95th percentile:        {distances['p95_distance']:.4f}")
        print(f"  99th percentile:        {distances['p99_distance']:.4f}")
        print(f"  Max distance:           {distances['max_distance']:.4f}")
        print(f"  Active tokens:          {distances['num_active']:,}")
        
        # Interpretation
        if usage['entropy_ratio'] < 0.5:
            print("\n  ⚠️  WARNING: Low entropy ratio suggests mode collapse!")
        if distances['mean_distance_mispredicted'] > distances['mean_distance'] * 1.5:
            print("\n  ⚠️  WARNING: Mispredicted tokens are far from GT (poor quality)")
        if distances['exact_match_rate'] < 0.3:
            print("\n  ⚠️  WARNING: Low exact match rate - predictions are often wrong")
    
    print("\n" + "=" * 80)
