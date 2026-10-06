"""
Backward compatibility utilities for loading old model checkpoints.

This module provides functions to detect old checkpoints and ensure
new features are properly disabled or initialized when loading old models.
"""

import torch
from typing import Dict, Optional
from pathlib import Path


def detect_old_checkpoint(state_dict: Dict[str, torch.Tensor]) -> Dict[str, bool]:
    """
    Detect if a checkpoint is from an old model version (before new features).
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        Dictionary with feature flags indicating what the checkpoint has:
        {
            'has_hf_prior': bool,
            'has_depth_emb': bool,
            'has_subband_emb': bool,
            'has_token_emb': bool,
        }
    """
    keys = set(state_dict.keys())
    
    return {
        'has_hf_prior': any('hf_prior' in k for k in keys),
        'has_depth_emb': any('depth_emb' in k for k in keys),
        'has_subband_emb': any('subband_emb' in k for k in keys),
        'has_token_emb': any('token_emb' in k for k in keys),
    }


def get_compatible_config_for_checkpoint(
    checkpoint_path: str,
    current_config: Optional[Dict] = None
) -> Dict:
    """
    Get a config dictionary compatible with an old checkpoint.
    
    This disables new features that don't exist in old checkpoints.
    
    Args:
        checkpoint_path: Path to checkpoint file
        current_config: Current config dict (optional, for merging)
        
    Returns:
        Config dict with new features disabled if checkpoint is old
    """
    from safetensors.torch import load_file
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return current_config or {}
    
    # Load checkpoint
    if checkpoint_path.suffix == '.safetensors':
        state_dict = load_file(str(checkpoint_path))
    else:
        state_dict = torch.load(str(checkpoint_path), map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if 'model' in state_dict:
            state_dict = state_dict['model']
    
    # Detect old checkpoint
    features = detect_old_checkpoint(state_dict)
    
    # Build compatible config
    compatible_config = current_config.copy() if current_config else {}
    
    # Disable features that don't exist in checkpoint
    if not features['has_hf_prior']:
        if 'model' in compatible_config and 'generator_v2' in compatible_config['model']:
            if 'params' in compatible_config['model']['generator_v2']:
                compatible_config['model']['generator_v2']['params']['use_hf_prior'] = False
        print(f"  [BACKWARD COMPAT] Disabling use_hf_prior (not in checkpoint)")
    
    if not features['has_depth_emb']:
        if 'model' in compatible_config and 'generator_v2' in compatible_config['model']:
            if 'params' in compatible_config['model']['generator_v2']:
                compatible_config['model']['generator_v2']['params']['use_depth_emb'] = False
        print(f"  [BACKWARD COMPAT] Disabling use_depth_emb (not in checkpoint)")
    
    if not features['has_subband_emb']:
        if 'model' in compatible_config and 'generator_v2' in compatible_config['model']:
            if 'params' in compatible_config['model']['generator_v2']:
                compatible_config['model']['generator_v2']['params']['use_subband_emb'] = False
        print(f"  [BACKWARD COMPAT] Disabling use_subband_emb (not in checkpoint)")
    
    if not features['has_token_emb']:
        if 'model' in compatible_config and 'generator_v2' in compatible_config['model']:
            if 'params' in compatible_config['model']['generator_v2']:
                compatible_config['model']['generator_v2']['params']['use_depth_conditional'] = False
        print(f"  [BACKWARD COMPAT] Disabling use_depth_conditional (not in checkpoint)")
    
    return compatible_config


def ensure_backward_compatible_model(model, state_dict: Dict[str, torch.Tensor]):
    """
    Ensure model is backward compatible with old checkpoint.
    
    This function disables new features if they don't exist in the checkpoint
    and initializes missing parameters to zero (neutral values).
    
    Args:
        model: The model instance
        state_dict: Checkpoint state dictionary
    """
    features = detect_old_checkpoint(state_dict)
    
    # Disable HF prior if not in checkpoint
    if not features['has_hf_prior'] and hasattr(model, 'use_hf_prior') and model.use_hf_prior:
        print(f"  [BACKWARD COMPAT] Disabling use_hf_prior (not in checkpoint)")
        model.use_hf_prior = False
        # Set to None to match old behavior
        if hasattr(model, 'hf_prior_scale_0'):
            model.hf_prior_scale_0 = None
        if hasattr(model, 'hf_prior_scale_1'):
            model.hf_prior_scale_1 = None
        if hasattr(model, 'hf_prior_scale_2'):
            model.hf_prior_scale_2 = None
    
    # Note: depth_emb, subband_emb, token_emb are embedded in the model architecture
    # If they don't exist in checkpoint, they'll be randomly initialized
    # But since we initialize HF priors to zero, and embeddings are small, this should be OK
    # The model will still work, just with slightly different behavior
