# NOTE: This implementation requires the official Mamba package.
# You can install it by running: pip install mamba-ssm

import torch
import torch.nn as nn
from . import config

# Conditional import of Mamba
try:
    from mamba_ssm import Mamba
except ImportError:
    print("Mamba package not found. Please install it with \"pip install mamba-ssm\"")
    Mamba = None

class MambaModel(nn.Module):
    """
    A Mamba / State Space Model (SSM) for sequence generation.
    This model uses the official `mamba-ssm` package.
    """
    def __init__(self, vocab_size):
        """
        Initializes the Mamba model.

        Args:
            vocab_size (int): The number of unique tokens in the vocabulary.
        """
        super(MambaModel, self).__init__()

        if Mamba is None:
            raise ImportError("Mamba package is required to use this model.")

        self.vocab_size = vocab_size

        # --- Layers ---
        # Embedding layer to convert token IDs to vectors
        self.embedding = nn.Embedding(self.vocab_size, config.d_model)

        # Mamba blocks
        self.layers = nn.ModuleList([Mamba(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
            ) for _ in range(config.n_layer)])
        
        # Normalization layer
        self.norm = nn.LayerNorm(config.d_model)

        # Fully connected layer to map Mamba output to vocabulary size
        self.lm_head = nn.Linear(config.d_model, self.vocab_size, bias=False)

    def forward(self, input_ids):
        """
        Performs a forward pass through the model.

        Args:
            input_ids (torch.Tensor): The input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: The output logits of shape (batch_size, sequence_length, vocab_size).
        """
        # 1. Embed the input sequence
        x = self.embedding(input_ids)

        # 2. Pass through Mamba blocks
        for layer in self.layers:
            x = layer(x)

        # 3. Normalize the output
        x = self.norm(x)

        # 4. Get the logits
        logits = self.lm_head(x)

        return logits, None # Returning None for compatibility
