import torch
from torch import nn
import torchaudio


class ConformerDecoder(nn.Module):
    """Conformer-based decoder with constrained day-specific input blocks.

    This version shrinks the Conformer, adds normalization, and constrains
    day-specific transforms to reduce exploding gradients.
    """

    def __init__(
        self,
        neural_dim,
        n_units,
        n_days,
        n_classes,
        n_heads=4,
        n_layers=6,
        input_dropout=0.2,
        patch_size=0,
        patch_stride=0,
        conv_kernel_size=15,
        ffn_mult=2,
        add_day_layers=True,
        day_layer_mode="mlp",  # "mlp" (current), "rnn" (GRU-style), or "none"
    ):
        super().__init__()

        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.n_days = n_days
        self.input_dropout = input_dropout
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.add_day_layers = add_day_layers
        self.day_layer_mode = day_layer_mode

        # --- Day-Specific Layers (constrained) ---
        self.day_layers = None
        self.day_weights = None
        self.day_biases = None
        self.day_activation = None
        self.day_dropout = None

        if add_day_layers and day_layer_mode == "mlp":
            self.day_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(neural_dim, neural_dim, bias=True),
                        nn.LayerNorm(neural_dim),
                        nn.GELU(),
                        nn.Dropout(input_dropout),
                    )
                    for _ in range(n_days)
                ]
            )
            for layer in self.day_layers:
                nn.init.xavier_uniform_(layer[0].weight, gain=0.1)
                nn.init.zeros_(layer[0].bias)

        if add_day_layers and day_layer_mode == "rnn":
            # Match the successful RNN day-specific behavior: Softsign over per-day affine.
            self.day_activation = nn.Softsign()
            self.day_dropout = nn.Dropout(input_dropout)
            self.day_weights = nn.ParameterList(
                [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
            )
            self.day_biases = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
            )

        # --- Input projection (with optional patching) ---
        self.input_size = self.neural_dim if self.patch_size == 0 else self.neural_dim * self.patch_size
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_size, n_units),
            nn.LayerNorm(n_units),
            nn.Dropout(input_dropout),
        )

        # --- Conformer Encoder ---
        ffn_dim = int(n_units * ffn_mult)
        self.pre_norm = nn.LayerNorm(n_units)
        self.conformer = torchaudio.models.Conformer(
            input_dim=n_units,
            num_heads=n_heads,
            ffn_dim=ffn_dim,
            num_layers=n_layers,
            depthwise_conv_kernel_size=conv_kernel_size,
            dropout=input_dropout,
        )

        # --- Output Layer ---
        self.output_norm = nn.LayerNorm(n_units)
        self.output_dropout = nn.Dropout(0.1)
        self.out = nn.Linear(n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight, gain=0.1)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, day_idx, states=None, return_state=False):
        # Day-specific transform (per-sample selection)
        if self.add_day_layers:
            if self.day_layer_mode == "mlp" and self.day_layers is not None:
                x = torch.stack([self.day_layers[d](x[i]) for i, d in enumerate(day_idx)], dim=0)
            elif self.day_layer_mode == "rnn" and self.day_weights is not None:
                day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
                day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)
                x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
                x = self.day_activation(x)
                x = self.day_dropout(x)
            elif self.day_layer_mode == "none":
                pass
            else:
                raise ValueError(f"Unsupported day_layer_mode: {self.day_layer_mode}")

        # Optional temporal patching
        if self.patch_size > 0:
            # (B, T, C) -> (B, 1, T, C)
            x = x.unsqueeze(1).permute(0, 3, 1, 2)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)
            x_unfold = x_unfold.squeeze(2).permute(0, 2, 3, 1)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)

        # Project to Conformer dimension
        x = self.input_proj(x)

        # torchaudio Conformer requires lengths
        lengths = torch.full((x.size(0),), x.size(1), device=x.device)
        x = self.pre_norm(x)
        output, _ = self.conformer(x, lengths)

        output = self.output_norm(output)
        output = self.output_dropout(output)
        logits = self.out(output)

        if return_state:
            return logits, None  # Conformer is stateless
        return logits
