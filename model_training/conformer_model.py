import torch
from torch import nn
import torchaudio

class ConformerDecoder(nn.Module):
    def __init__(self,
                 neural_dim,
                 n_units, # Used as Conformer FFN dim
                 n_days,
                 n_classes,
                 n_heads=4, 
                 n_layers=6, 
                 input_dropout=0.0,
                 patch_size=0,
                 patch_stride=0,
                 ):
        super(ConformerDecoder, self).__init__()
        
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.n_days = n_days
        self.input_dropout = input_dropout
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # --- Day-Specific Layers (Same as RNN) ---
        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.ParameterList([nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)])
        self.day_biases = nn.ParameterList([nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)])
        self.day_layer_dropout = nn.Dropout(input_dropout)
        
        # Calculate input dimension
        self.input_size = self.neural_dim
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        # --- Conformer Encoder ---
        # Conformer requires input_dim to be divisible by n_heads, usually we project it
        self.projection = nn.Linear(self.input_size, n_units)
        
        self.conformer = torchaudio.models.Conformer(
            input_dim=n_units,
            num_heads=n_heads,
            ffn_dim=n_units * 4, 
            num_layers=n_layers,
            depthwise_conv_kernel_size=31,
            dropout=input_dropout
        )

        # --- Output Layer ---
        self.out = nn.Linear(n_units, self.n_classes)

    def forward(self, x, day_idx, states=None, return_state=False):
        # 1. Day-specific projection
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)
        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # 2. Patching (if configured)
        if self.patch_size > 0: 
            x = x.unsqueeze(1).permute(0, 3, 1, 2)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)
            x_unfold = x_unfold.squeeze(2).permute(0, 2, 3, 1)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)

        # 3. Conformer Forward
        x = self.projection(x)
        # torchaudio Conformer requires lengths parameter, assuming full length here
        lengths = torch.full((x.size(0),), x.size(1), device=x.device)
        output, _ = self.conformer(x, lengths)

        # 4. Classification
        logits = self.out(output)
        
        if return_state:
            return logits, None # Conformer is stateless
        return logits
