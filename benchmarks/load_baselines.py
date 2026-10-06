import os
import torch

def load_swinir(weights_path: str, device: str = 'cuda'):
    """
    Loads the SwinIR baseline wrapper.
    """
    from benchmarks.swinir.train_swinir import SwinIRWrapper
    swinir_model = SwinIRWrapper(
        n_colors=1,
        embed_dim=256,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[8, 8, 8, 8, 8, 8],
        window_size=8,
        mlp_ratio=2.0,
    )
    num_trainable = sum(p.numel() for p in swinir_model.parameters() if p.requires_grad)
    print(f" [SwinIR] Number of trainable parameters: {num_trainable:,}")
    
    weights = torch.load(weights_path, map_location='cpu')
    swinir_model.load_state_dict(weights)
    swinir_model.to(device)
    swinir_model.eval()
    return swinir_model


def load_flow(weights_path: str, device: str = 'cuda'):
    """
    Loads the Fast Diffusion / Flow Matching baseline wrapper.
    """
    from benchmarks.fast_diffusion.train_fast_diffusion import FastDiffusionWrapper
    flow_model = FastDiffusionWrapper(n_colors=1, image_size=48, num_inference_steps=4)
    num_trainable = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
    print(f" [Flow Matching] Number of trainable parameters: {num_trainable:,}")
    
    flow_weights = torch.load(weights_path, map_location='cpu')
    flow_model.load_state_dict(flow_weights)
    flow_model.to(device)
    flow_model.eval()
    return flow_model


def load_sr3(weights_path: str, device: str = 'cuda'):
    """
    Loads the SR3 Diffusion baseline wrapper.
    """
    from benchmarks.sr3.train_sr3 import SR3Wrapper
    sr3_model = SR3Wrapper()
    num_trainable = sum(p.numel() for p in sr3_model.parameters() if p.requires_grad)
    print(f" [SR3] Number of trainable parameters: {num_trainable:,}")
    
    weights = torch.load(weights_path, map_location='cpu')
    sr3_model.load_state_dict(weights)
    sr3_model.to(device)
    sr3_model.eval()
    return sr3_model


def load_varsr(varsr_weights_path: str, var_tokenizer_weights_path: str, device: str = 'cuda'):
    """
    Loads the VARSR baseline wrapper along with its VQVAE tokenizer.
    """
    from benchmarks.varsr.wrapper import VARSRWrapper
    from benchmarks.varsr.varsr_repo_imports import make_lidar_varsr_vqvae

    vae = make_lidar_varsr_vqvae()
    vae_ckpt = torch.load(var_tokenizer_weights_path, map_location="cpu")
    if isinstance(vae_ckpt, dict) and "vqvae" in vae_ckpt:
        vae.load_state_dict(vae_ckpt["vqvae"])
    else:
        state_dict = vae_ckpt
        if isinstance(state_dict, dict) and any(k.startswith("vqvae.") for k in state_dict.keys()):
            state_dict = {k.replace("vqvae.", ""): v for k, v in state_dict.items() if k.startswith("vqvae.")}
        vae.load_state_dict(state_dict)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    num_trainable = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f" [VAR Tokenizer] Number of trainable parameters: {num_trainable:,}")

    varsr_model = VARSRWrapper(
        vae_local=vae,
        embed_dim=256,
        depth=12,
        patch_nums=(1, 2, 3, 4, 6, 12),
        num_heads=4,
        mlp_ratio=4
    )
    num_trainable = sum(p.numel() for p in varsr_model.parameters() if p.requires_grad)
    print(f" [VARSR] Number of trainable parameters: {num_trainable:,}")

    varsr_weights = torch.load(varsr_weights_path, map_location='cpu')
    varsr_model.load_state_dict(varsr_weights)
    varsr_model.to(device)
    varsr_model.eval()
    return varsr_model
