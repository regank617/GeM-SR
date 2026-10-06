import os
import torch
from safetensors.torch import load_file
from omegaconf import OmegaConf

def load_gemsr(
    ssm_config_path: str, 
    ssm_ckpt_path: str, 
    tokenizer_config_path: str, 
    tokenizer_ckpt_path: str,
    device: str = 'cuda'
):
    """
    Loads the GeM-SR model and its conditioner from the given configuration and checkpoint paths.
    
    Args:
        ssm_config_path: Path to the generator's config.yaml
        ssm_ckpt_path: Path to the generator's model.safetensors
        tokenizer_config_path: Path to the tokenizer's config.yaml
        tokenizer_ckpt_path: Path to the tokenizer's model.safetensors
        device: Device to load the models onto
    """
    from src.modeling.generator.conditioner import extract_conditioner
    from training.train_generator import build_model_v5
    
    config = OmegaConf.load(ssm_config_path)
    config.model.tokenizer.pretrained.path_to_config = tokenizer_config_path
    config.model.tokenizer.pretrained.path_to_ckpt = tokenizer_ckpt_path

    ssm, pretrainer = build_model_v5(config, device=device, is_resume=False)
    num_trainable = sum(p.numel() for p in ssm.parameters() if p.requires_grad)
    print(f" [GeM-SR] Number of trainable parameters: {num_trainable:,}")

    ssm_weights = load_file(ssm_ckpt_path)
    
    # The conditioner weights are usually saved as 'model_1.safetensors' alongside the generator
    cond_ckpt_path = os.path.join(os.path.dirname(ssm_ckpt_path), 'model_1.safetensors')
    if os.path.exists(cond_ckpt_path):
        cond_weights = load_file(cond_ckpt_path)
        pretrainer.load_state_dict(cond_weights)
        print('Loaded conditioner weights')
    else:
        print(f"Warning: Conditioner weights not found at {cond_ckpt_path}")

    ssm.load_state_dict(ssm_weights)
    print('Loaded GeM-SR weights')
    
    conditioner = extract_conditioner(pretrainer)
    return ssm, conditioner


def get_gemsr_results(
    ssm, 
    conditioner, 
    lr, 
    temps={'root': 1.0, 'scale_0': 1.0, 'scale_1': 0.8, 'scale_2': 0.8}, 
    top_k=50, 
    top_p=0.85, 
    cfg_scale=1.0,
    device='cuda'
):
    """
    Performs inference using the loaded GeM-SR model and conditioner.
    
    Args:
        ssm: The generator model
        conditioner: The conditioner model
        lr: Low-resolution input tensor [B, C, H, W]
        temps: Temperature scaling dictionary for each hierarchical scale
        top_k: Top-K sampling cutoff
        top_p: Nucleus sampling probability cutoff
        cfg_scale: Classifier-Free Guidance scale (if applicable)
        device: Device to run inference on
    
    Returns:
        recon: The final super-resolved CHM
        prerefiner_recon: The reconstruction prior to the final refiner layer
    """
    ssm.to(device)
    ssm.eval()
    conditioner.to(device)
    conditioner.eval()
    
    with torch.no_grad():
        cond_outputs = conditioner(lr.to(device))
        output = ssm.generate(
            cond_outputs, 
            ssm.tokenizer,
            top_k=top_k,
            top_p=top_p,
            temps=temps,
            cfg_scale=cfg_scale
        )

        root = ssm.tokenizer.decode_root(output['root'])
        s0 = ssm.tokenizer.inverse_step(root, output['scale_0'], scale_idx=0)
        s1 = ssm.tokenizer.inverse_step(s0, output['scale_1'], scale_idx=1)
        s2 = ssm.tokenizer.inverse_step(s1, output['scale_2'], scale_idx=2)
        
        recon = ssm.tokenizer.features_to_pixels(s2)
        _, prerefiner_recon = ssm.tokenizer.features_to_pixels(s2, ret_prerefiner=True)
        
    return recon, prerefiner_recon
