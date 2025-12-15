import torch
import h5py
import os
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import sys

# Ensure we can import local modules
sys.path.append(os.getcwd())
from unet_model import NeuralUNet

def preprocess_session(session_dir, filename, model, device, batch_size=32):
    src_path = os.path.join(session_dir, filename)
    dst_path = os.path.join(session_dir, filename.replace('.hdf5', '_unet.hdf5'))
    
    if not os.path.exists(src_path):
        print(f"Skipping {src_path}, not found.")
        return

    if os.path.exists(dst_path):
        print(f"Skipping {dst_path}, already exists.")
        return

    print(f"Processing {src_path} -> {dst_path}")
    
    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        # Copy attributes
        for k, v in src.attrs.items():
            dst.attrs[k] = v
            
        # Copy all datasets first (labels, etc.)
        for key in src.keys():
            if key != 'input_features':
                src.copy(key, dst)
        
        # Process input_features
        if 'input_features' in src:
            dset_in = src['input_features']
            shape = dset_in.shape
            # Create dataset in dst
            dset_out = dst.create_dataset('input_features', shape=shape, dtype=np.float32)
            
            # Copy attributes for input_features
            for k, v in dset_in.attrs.items():
                dset_out.attrs[k] = v

            # Process in batches to save memory
            model.eval()
            with torch.no_grad():
                for i in tqdm(range(0, shape[0], batch_size), desc=f"  {filename}", leave=False):
                    # Load batch
                    batch_data = dset_in[i:i+batch_size]
                    batch_tensor = torch.from_numpy(batch_data).to(device).float()
                    
                    # Prepare for U-Net: (B, T, C) -> (B, 1, T, C)
                    x_in = batch_tensor.unsqueeze(1)
                    
                    # Forward pass (Denoising/Feature Extraction)
                    x_enhanced = model(x_in)
                    
                    # Back to original shape: (B, 1, T, C) -> (B, T, C)
                    x_out = x_enhanced.squeeze(1)
                    
                    # Save to destination
                    dset_out[i:i+batch_size] = x_out.cpu().numpy()

def main():
    # Load configuration
    args = OmegaConf.load('rnn_args.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load U-Net
    unet_path = args.model.unet_path
    if not unet_path or not os.path.exists(unet_path):
        raise ValueError(f"U-Net path not found: {unet_path}")

    print(f"Loading U-Net from {unet_path}")
    # Initialize U-Net (assuming 1 channel input/output as per combined_models.py)
    model = NeuralUNet(n_channels=1, n_classes=1)
    
    checkpoint = torch.load(unet_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    # Dataset paths
    dataset_dir = args.dataset.dataset_dir
    sessions = args.dataset.sessions
    
    # Process all sessions
    for session in tqdm(sessions, desc="Total Progress"):
        session_dir = os.path.join(dataset_dir, session)
        preprocess_session(session_dir, 'data_train.hdf5', model, device)
        preprocess_session(session_dir, 'data_val.hdf5', model, device)

    print("\nPre-processing complete! You can now train with use_precomputed_unet: true")

if __name__ == '__main__':
    main()
