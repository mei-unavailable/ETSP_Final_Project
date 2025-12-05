import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import os
import datetime
import numpy as np
import random
from tqdm import tqdm
import logging
import sys
import pathlib

from dataset import BrainToTextDataset, train_test_split_indicies
from unet_model import NeuralUNet

def get_mask(data, mask_ratio=0.5, device='cpu'):
    """
    Generate random mask
    data: (Batch, Time, Channels)
    Returns:
        mask: (Batch, 1, Time, Channels) binary mask, 1 means masked (hidden), 0 means visible
    """
    B, T, C = data.shape
    # Generate random probability matrix
    prob = torch.rand(B, T, C, device=device)
    # Create mask: probability < mask_ratio are masked (set to 1)
    mask = (prob < mask_ratio).float()
    return mask.unsqueeze(1) # Add channel dim for UNet: (B, 1, T, C)

def train_ssl():
    # 1. Configuration
    args = OmegaConf.load('rnn_args.yaml')
    
    # Setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # Device setup
    if torch.cuda.is_available():
        gpu_num = args.get('gpu_number', 0)
        try:
            gpu_num = int(gpu_num)
            device = torch.device(f"cuda:{gpu_num}")
        except:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")

    # Hyperparameters
    MASK_RATIO = 0.5
    BATCH_SIZE = 16
    LR = 1e-4
    EPOCHS = 50
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_DIR = f'trained_models/unet_ssl_{timestamp}'
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 2. Dataset
    # Create datasets and dataloaders (Logic copied from rnn_trainer.py)
    train_file_paths = [os.path.join(args["dataset"]["dataset_dir"],s,'data_train.hdf5') for s in args['dataset']['sessions']]
    
    # Split trials into train and test sets
    train_trials, _ = train_test_split_indicies(
        file_paths = train_file_paths, 
        test_percentage = 0,
        seed = args['dataset']['seed'],
        bad_trials_dict = None,
        )

    # Determine if a only a subset of neural features should be used
    feature_subset = None
    if ('feature_subset' in args['dataset']) and args['dataset']['feature_subset'] != None: 
        feature_subset = args['dataset']['feature_subset']
        
    # train dataset and dataloader
    train_dataset = BrainToTextDataset(
        trial_indicies = train_trials,
        split = 'train',
        days_per_batch = args['dataset']['days_per_batch'],
        n_batches = 1000, # Arbitrary number of batches per epoch
        batch_size = BATCH_SIZE,
        must_include_days = None,
        random_seed = args['dataset']['seed'],
        feature_subset = feature_subset
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size = None, # Dataset.__getitem__() already returns batches
        shuffle = True,
        num_workers = 4,
        pin_memory = True 
    )

    # 3. Model
    model = NeuralUNet(n_channels=1, n_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss(reduction='none') 

    logger.info("Starting Self-Supervised Pre-training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            original_input = batch['input_features'].to(device) # (B, T, C)
            valid_lens = batch['n_time_steps'].to(device)
            
            # Prepare input: (B, 1, T, C)
            x = original_input.unsqueeze(1)
            
            # Generate mask
            mask = get_mask(original_input, MASK_RATIO, device)
            
            # Apply mask: masked areas set to 0
            masked_input = x * (1 - mask)
            
            # Forward
            reconstruction = model(masked_input)
            
            # Calculate Loss
            loss = criterion(reconstruction, x)
            
            # MAE Core: Only calculate loss on masked parts
            # Also ignore padding
            B, _, T, C = x.shape
            padding_mask = torch.zeros_like(mask)
            for i, length in enumerate(valid_lens):
                padding_mask[i, :, :length, :] = 1.0
            
            # Final Loss mask = Masked Part AND Valid (Non-Padding) Part
            final_loss_mask = mask * padding_mask
            
            loss = (loss * final_loss_mask).sum() / (final_loss_mask.sum() + 1e-6)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': total_loss / num_batches})

        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.6f}")
        
        # Save model
        torch.save(model.state_dict(), f"{SAVE_DIR}/unet_mae_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train_ssl()
