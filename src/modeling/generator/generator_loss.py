"""
Hierarchical Loss Functions for Structure vs Content Separation

This module implements the loss design specified in the V2 refactor:
- Feature loss: Cross-entropy on tokens ONLY at structure positions (first_zero_depth > 0).
  Within those positions, loss is computed for depths 0..first_null (inclusive):
  features + the single EOS token. Spatial zeros (first_zero_depth == 0) get NO
  token CE; existence loss alone drives "no structure here."
- Existence loss: Binary cross-entropy on all positions (primary sparsity driver).
- Density regularization: Weak L1-style regularization on predicted vs GT density.

Key principle: Existence head = WHERE structure is; feature CE = WHAT token only
where structure exists. Zero tokens at void positions are not trained via CE;
they are assigned at inference by hard-shunting from existence. EOS (one zero
per structure position at first_zero_depth) is still trained in token CE so the
model learns when to stop (VBR).
"""


import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


def _fsq_null_to_int(z) -> int:
    """FSQ stores ``zero_idx`` as a buffer tensor; configs may pass int. Never use ``tensor == None`` (that is a Python bool)."""
    if z is None:
        raise ValueError("FSQ null index is None; pass tokenizer= or explicit zero_indices_dict / zero_idx.")
    if isinstance(z, torch.Tensor):
        return int(z.detach().cpu().item())
    return int(z)


def null_indices_dict_from_tokenizer(tokenizer) -> Dict[str, int]:
    """
    Map ``root`` / ``scale_0`` / ``scale_1`` / ``scale_2`` to the FSQ void index used in targets and logits.

    **Separate detail codebooks** (e.g. ``LidarTokenizerV6``): one null index per scale from
    ``null_indices_det`` or, if missing, ``[fsq.zero_idx for fsq in fsq_details_list]``.

    **Joint detail codebook** (e.g. ``tokenizerV5_old.LidarTokenizerV5``): a single detail null
    ``null_idx_det`` (shared ``fsq_details``) is used for ``scale_0`` … ``scale_2``; root still uses
    ``fsq_root`` via ``null_idx_root``.
    """
    if not hasattr(tokenizer, "null_idx_root"):
        raise AttributeError(
            f"Expected tokenizer with null_idx_root; got {type(tokenizer).__name__}"
        )
    out: Dict[str, int] = {"root": _fsq_null_to_int(tokenizer.null_idx_root)}

    detail_nulls: Optional[List] = None
    if hasattr(tokenizer, "null_indices_det") and tokenizer.null_indices_det is not None:
        detail_nulls = list(tokenizer.null_indices_det)
    elif hasattr(tokenizer, "fsq_details_list") and tokenizer.fsq_details_list is not None:
        n = len(tokenizer.fsq_details_list)
        if n >= 3:
            detail_nulls = [fsq.zero_idx for fsq in tokenizer.fsq_details_list[:3]]

    if detail_nulls is not None:
        for i, name in enumerate(["scale_0", "scale_1", "scale_2"]):
            if i >= len(detail_nulls):
                raise ValueError(
                    f"Need 3 detail null indices for scale_0..2; got {len(detail_nulls)} "
                    f"from {type(tokenizer).__name__}"
                )
            out[name] = _fsq_null_to_int(detail_nulls[i])
        return out

    if hasattr(tokenizer, "null_idx_det"):
        zi = _fsq_null_to_int(tokenizer.null_idx_det)
        for name in ["scale_0", "scale_1", "scale_2"]:
            out[name] = zi
        return out

    raise AttributeError(
        f"{type(tokenizer).__name__}: expected null_indices_det / fsq_details_list (separate) "
        f"or null_idx_det (joint detail FSQ)."
    )


def compute_hierarchical_loss_v5(
    logits,
    target_indices,
    zero_idx: int,
    *,
    use_ce_weighting: bool = True,
    void_class_weight: float = 0.5,
):
    """
    Computes Cross Entropy and a full suite of health diagnostics for a single scale.
    logits: [B, 1, H, W, V]
    target_indices: [B, 1, H, W]
    zero_idx: Python int void index for this scale's vocabulary (must not be None).
    """
    zi = int(zero_idx)  # Ensure it's a standard python int
    
    # 1. Flatten for metric computation
    flat_logits = logits.flatten(0, 3)          # [N, V]
    flat_targets = target_indices.flatten(0, 3) # [N]
    
    vocab_size = flat_logits.shape[-1]
    
    class_weights = torch.ones(vocab_size, device=logits.device, dtype=flat_logits.dtype)
    if use_ce_weighting:
        class_weights[zi] = float(void_class_weight)

    loss = F.cross_entropy(flat_logits, flat_targets, weight=class_weights)
    
    # ==========================================
    # 3. DIAGNOSTICS SUITE (No Gradients Needed)
    # ==========================================
    with torch.no_grad():
        predictions = flat_logits.argmax(dim=-1)
        
        is_correct = (predictions == flat_targets)
        overall_acc = is_correct.float().mean().item()
        
        is_gt_zero = (flat_targets == zi)
        is_gt_active = ~is_gt_zero
        
        void_acc = is_correct[is_gt_zero].float().mean().item() if bool(is_gt_zero.any()) else 0.0
        active_acc = is_correct[is_gt_active].float().mean().item() if bool(is_gt_active.any()) else 0.0
            
        gt_sparsity = is_gt_zero.float().mean().item()
        pred_sparsity = (predictions == zi).float().mean().item()
        
        probs = F.softmax(flat_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()
        
        stats = {
            'loss': loss.item(),
            'acc_overall': overall_acc * 100.0,
            'acc_void': void_acc * 100.0,
            'acc_active': active_acc * 100.0,
            'gt_sparsity': gt_sparsity * 100.0,
            'pred_sparsity': pred_sparsity * 100.0,
            'entropy': entropy
        }
        
    return loss, stats


def compute_multi_scale_loss_v5(
    scale_outputs,
    scale_targets,
    zero_idx=None,
    tokenizer=None,
    zero_indices_dict: Optional[Dict[str, int]] = None,
    scale_loss_weights: Optional[Dict[str, float]] = None,
    *,
    use_ce_weighting: bool = True,
    void_class_weight: float = 1.0, # only for FSQ research, not used for models in paper
):
    """
    Computes the weighted sum of losses across all scales.
    Now defaults to emphasizing fine scales (scale_1 and scale_2) to fight sparsity.
    """
    if zero_indices_dict is None:
        if tokenizer is not None:
            zero_indices_dict = null_indices_dict_from_tokenizer(tokenizer)
        elif zero_idx is not None:
            zi = int(zero_idx)
            zero_indices_dict = {n: zi for n in ["root", "scale_0", "scale_1", "scale_2"]}
        else:
            raise ValueError(
                "compute_multi_scale_loss_v5 needs tokenizer=, zero_indices_dict=, or zero_idx= (int)."
            )

    scale_names = ["root", "scale_0", "scale_1", "scale_2"]
    
    # ==========================================
    # THE FIX: BOOSTING FINE-SCALE GRADIENTS
    # ==========================================
    if scale_loss_weights is None:
        # Default to heavy fine-scale weighting if nothing is passed from config
        weights = {'root': 1.0, 'scale_0': 1.0, 'scale_1': 1.5, 'scale_2': 2.0}
    else:
        weights = {n: 1.0 for n in scale_names}
        for n in scale_names:
            if n in scale_loss_weights and scale_loss_weights[n] is not None:
                weights[n] = float(scale_loss_weights[n])
                
    w_sum = sum(weights.values())
    if w_sum <= 0:
        raise ValueError("scale_loss_weights must sum to a positive value.")

    total_weighted = torch.zeros((), device=scale_outputs["root"].device, dtype=scale_outputs["root"].dtype)
    metrics = {}

    for name in scale_names:
        loss, stats = compute_hierarchical_loss_v5(
            scale_outputs[name],
            scale_targets[name],
            zero_indices_dict[name],
            use_ce_weighting=use_ce_weighting,
            void_class_weight=void_class_weight,
        )
        w = weights[name]
        total_weighted = total_weighted + w * loss
        for k, v in stats.items():
            metrics[f"{name}/{k}"] = v

    mean_loss = total_weighted / w_sum
    metrics["total_loss"] = mean_loss.item()
    for n in scale_names:
        metrics[f"loss_weight/{n}"] = weights[n] / w_sum
        
    return mean_loss, metrics