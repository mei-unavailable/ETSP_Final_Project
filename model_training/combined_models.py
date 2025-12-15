import torch
import torch.nn as nn
import os
from unet_model import NeuralUNet
from rnn_model import GRUDecoder
from conformer_model import ConformerDecoder

class UNetEnhancedModel(nn.Module):
    """
    A combined model that uses a pre-trained U-Net as a feature extractor/denoiser
    before passing the data to a downstream decoder (RNN or Conformer).
    """
    def __init__(self, 
                 base_model_type, 
                 base_model_args, 
                 unet_path=None, 
                 freeze_unet=True):
        """
        Args:
            base_model_type (str): 'rnn' or 'conformer'
            base_model_args (dict): Arguments to initialize the base model
            unet_path (str): Path to the pre-trained U-Net checkpoint
            freeze_unet (bool): Whether to freeze the U-Net parameters
        """
        super(UNetEnhancedModel, self).__init__()
        
        # 1. Initialize U-Net Feature Extractor
        # Note: NeuralUNet input/output channels are 1 for this specific neural data shape (B, 1, T, C)
        self.unet = NeuralUNet(n_channels=1, n_classes=1)
        
        if unet_path:
            if os.path.exists(unet_path):
                print(f"Loading U-Net weights from {unet_path}")
                checkpoint = torch.load(unet_path, map_location='cpu')
                
                # Handle both full checkpoint dict and direct state_dict
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                    
                self.unet.load_state_dict(state_dict)
            else:
                print(f"Warning: U-Net path {unet_path} not found. Initializing with random weights.")
        
        # Optional: Freeze U-Net to use it purely as a fixed feature extractor
        if freeze_unet:
            for param in self.unet.parameters():
                param.requires_grad = False
            print("U-Net parameters frozen.")
        else:
            print("U-Net parameters trainable (fine-tuning).")

        # 2. Initialize Downstream Decoder
        if base_model_type == 'rnn':
            self.decoder = GRUDecoder(**base_model_args)
        elif base_model_type == 'conformer':
            self.decoder = ConformerDecoder(**base_model_args)
        else:
            raise ValueError(f"Unknown base_model_type: {base_model_type}")

    def forward(self, x, day_idx, states=None, return_state=False):
        """
        Args:
            x: (Batch, Time, Channels)
            day_idx: (Batch,)
        """
        # 1. U-Net Feature Extraction
        # U-Net expects (Batch, Channels, Time, Feats) -> (B, 1, T, C)
        x_in = x.unsqueeze(1) 
        
        # Pass through U-Net
        # The U-Net is trained to reconstruct/denoise, so output is same shape
        x_enhanced = self.unet(x_in)
        
        # Remove channel dim: (B, 1, T, C) -> (B, T, C)
        x_out = x_enhanced.squeeze(1)
        
        # Optional: Residual connection (Original + Enhanced)
        # x_out = x + x_out 

        # 2. Decoder Forward
        return self.decoder(x_out, day_idx, states, return_state)

def build_unet_conformer(conformer_args, unet_path=None, freeze_unet=False):
    """
    Helper function to build a Conformer initialized with U-Net weights (as frontend).
    This satisfies the request to 'feed U-Net weights as initial state to Conformer'.
    """
    return UNetEnhancedModel(
        base_model_type='conformer',
        base_model_args=conformer_args,
        unet_path=unet_path,
        freeze_unet=freeze_unet
    )

def build_unet_rnn(rnn_args, unet_path=None, freeze_unet=False):
    """
    Helper function to build an RNN initialized with U-Net weights (as frontend).
    """
    return UNetEnhancedModel(
        base_model_type='rnn',
        base_model_args=rnn_args,
        unet_path=unet_path,
        freeze_unet=freeze_unet
    )
