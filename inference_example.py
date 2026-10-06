"""
GeM-SR Inference Example
========================
This script demonstrates how to load a trained GeM-SR model (or baseline), 
perform inference on a batch of LiDAR CHM data, and compute basic evaluation metrics.

If you want a more comprehensive evaluation across the entire test set (with LPIPS, FID, etc.), 
please use `benchmark_hub.py`.
"""

import os
import sys

# Ensure the project root is in the path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import torch.nn.functional as F

# Import our unified loader functions
from src.modeling.generator.inference import load_gemsr, get_gemsr_results
from benchmarks.load_baselines import load_swinir, load_varsr, load_sr3, load_flow
from benchmarks.eval_wrappers import VARSREvalWrapper, SR3EvalWrapper, FastDiffusionEvalWrapper

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running inference on: {device}")

    # 1. Load the sample batch data
    # ylr: Low-Resolution input (e.g., 64x1x32x16)
    # xhr: High-Resolution target (e.g., 64x1x48x48)
    ylr_path = 'samples/test_batch/ylr_batch.npy'
    xhr_path = 'samples/test_batch/xhr_batch.npy'
    
    if not os.path.exists(ylr_path) or not os.path.exists(xhr_path):
        print(f"Error: Sample data not found at {ylr_path} or {xhr_path}.")
        print("Please ensure you are running this from the repository root.")
        return

    # Load numpy arrays and convert to PyTorch tensors
    ylr_batch = torch.from_numpy(np.load(ylr_path)).float().to(device)
    xhr_batch = torch.from_numpy(np.load(xhr_path)).float().to(device)
    
    print(f"Loaded LR input shape: {ylr_batch.shape}")
    print(f"Loaded HR target shape: {xhr_batch.shape}")

    # =========================================================================
    # Option A: Inference with GeM-SR
    # =========================================================================
    print("\n--- Testing GeM-SR ---")
    
    # NOTE: To run this locally, you MUST update these paths to point to your 
    # actual trained weights and configs.
    gemsr_config_path = "configs/generator/lidar_ssmVQ.yaml"
    gemsr_ckpt_path = "path/to/your/generator/model.safetensors"
    tokenizer_config_path = "configs/tokenizer/lidar_tokenizerVQ.yaml"
    tokenizer_ckpt_path = "path/to/your/tokenizer/model.safetensors"
    
    if os.path.exists(gemsr_ckpt_path) and os.path.exists(tokenizer_ckpt_path):
        print("Loading GeM-SR model...")
        gemsr, conditioner = load_gemsr(
            ssm_config_path=gemsr_config_path,
            ssm_ckpt_path=gemsr_ckpt_path,
            tokenizer_config_path=tokenizer_config_path,
            tokenizer_ckpt_path=tokenizer_ckpt_path,
            device=device
        )
        
        print("Running GeM-SR inference...")
        # get_gemsr_results takes the generator, conditioner, and LR input.
        # It handles the conditioning, token generation, and decoding automatically.
        recon, _ = get_gemsr_results(gemsr, conditioner, ylr_batch, device=device)
        
        # Calculate a simple metric (Mean Squared Error)
        mse_loss = F.mse_loss(torch.clamp(recon, 0.0, 1.0), torch.clamp(xhr_batch, 0.0, 1.0))
        print(f"GeM-SR MSE: {mse_loss.item():.4f}")
    else:
        print(f"Skipping GeM-SR: Weights not found at '{gemsr_ckpt_path}' or '{tokenizer_ckpt_path}'.")
        print("Please update the paths in this script to point to your trained checkpoints.")


    # =========================================================================
    # Option B: Inference with a Baseline (e.g., SwinIR)
    # =========================================================================
    print("\n--- Testing SwinIR Baseline ---")
    swinir_weights_path = "path/to/your/swinir/x.pth"
    
    if os.path.exists(swinir_weights_path):
        print("Loading SwinIR model...")
        swinir = load_swinir(weights_path=swinir_weights_path, device=device)
        
        print("Running SwinIR inference...")
        with torch.no_grad():
            recon_swinir = swinir(ylr_batch)
            
        mse_loss = F.mse_loss(torch.clamp(recon_swinir, 0.0, 1.0), torch.clamp(xhr_batch, 0.0, 1.0))
        print(f"SwinIR MSE: {mse_loss.item():.4f}")
    else:
        print(f"Skipping SwinIR: Weights not found at '{swinir_weights_path}'.")
        print("Please update the path to test SwinIR.")
        
    # =========================================================================
    # Option C: Inference with SR3 (DDPM & DDIM)
    # =========================================================================
    print("\n--- Testing SR3 Baseline ---")
    sr3_weights_path = "path/to/your/SR3/x.pth"
    
    if os.path.exists(sr3_weights_path):
        print("Loading SR3 model...")
        sr3_base = load_sr3(weights_path=sr3_weights_path, device=device)
        
        # 1. SR3 with standard DDPM
        print("Running SR3 inference (DDPM)...")
        sr3_ddpm = SR3EvalWrapper(sr3_base, ddim_steps=0).to(device)
        with torch.no_grad():
            recon_sr3_ddpm = sr3_ddpm(ylr_batch)
        mse_sr3_ddpm = F.mse_loss(torch.clamp(recon_sr3_ddpm, 0.0, 1.0), torch.clamp(xhr_batch, 0.0, 1.0))
        print(f"SR3 (DDPM) MSE: {mse_sr3_ddpm.item():.4f}")

        # 2. SR3 with DDIM (e.g. 50 steps)
        print("Running SR3 inference (DDIM - 50 steps)...")
        sr3_ddim = SR3EvalWrapper(sr3_base, ddim_steps=50).to(device)
        with torch.no_grad():
            recon_sr3_ddim = sr3_ddim(ylr_batch)
        mse_sr3_ddim = F.mse_loss(torch.clamp(recon_sr3_ddim, 0.0, 1.0), torch.clamp(xhr_batch, 0.0, 1.0))
        print(f"SR3 (DDIM 50) MSE: {mse_sr3_ddim.item():.4f}")
    else:
        print(f"Skipping SR3: Weights not found at '{sr3_weights_path}'.")

    # =========================================================================
    # Option D: Inference with Fast Diffusion (Flow Matching)
    # =========================================================================
    print("\n--- Testing Fast Diffusion (Flow Matching) Baseline ---")
    flow_weights_path = "path/to/your/flow/x.pth"
    
    if os.path.exists(flow_weights_path):
        print("Loading Fast Diffusion model...")
        flow_base = load_flow(weights_path=flow_weights_path, device=device)
        
        flow_eval = FastDiffusionEvalWrapper(flow_base, num_steps=4).to(device)
        
        print("Running Fast Diffusion inference...")
        with torch.no_grad():
            recon_flow = flow_eval(ylr_batch)
        mse_flow = F.mse_loss(torch.clamp(recon_flow, 0.0, 1.0), torch.clamp(xhr_batch, 0.0, 1.0))
        print(f"Fast Diffusion MSE: {mse_flow.item():.4f}")
    else:
        print(f"Skipping Fast Diffusion: Weights not found at '{flow_weights_path}'.")

    # =========================================================================
    # Option E: Inference with VARSR
    # =========================================================================
    print("\n--- Testing VARSR Baseline ---")
    varsr_weights_path = "path/to/your/varsr/var_step_x.pth"
    varsr_tokenizer_path = "path/to/your/var/tokenizer/x.pth"
    
    if os.path.exists(varsr_weights_path) and os.path.exists(varsr_tokenizer_path):
        print("Loading VARSR model...")
        varsr_base = load_varsr(
            varsr_weights_path=varsr_weights_path, 
            var_tokenizer_weights_path=varsr_tokenizer_path,
            device=device
        )
        # Paper / benchmark_hub.py sampling settings for VARSR.
        varsr_eval = VARSREvalWrapper(varsr_base, cfg=6.0, top_k=900, top_p=0.95).to(device)
        
        print("Running VARSR inference...")
        with torch.no_grad():
            recon_varsr = varsr_eval(ylr_batch)
        mse_varsr = F.mse_loss(torch.clamp(recon_varsr, 0.0, 1.0), torch.clamp(xhr_batch, 0.0, 1.0))
        print(f"VARSR MSE: {mse_varsr.item():.4f}")
    else:
        print(f"Skipping VARSR: Weights not found at '{varsr_weights_path}' or '{varsr_tokenizer_path}'.")

    print("\nInference tutorial complete! Feel free to modify this script to test other models.")

if __name__ == '__main__':
    main()
