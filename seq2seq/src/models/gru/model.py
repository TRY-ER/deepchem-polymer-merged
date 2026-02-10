import torch
import torch.nn as nn
from . import config

class GRUModel(nn.Module):
    """
    A Gated Recurrent Unit (GRU) model for sequence generation.
    """
    def __init__(self, vocab_size):
        """
        Initializes the GRU model.

        Args:
            vocab_size (int): The number of unique tokens in the vocabulary.
        """
        super(GRUModel, self).__init__()
        self.vocab_size = vocab_size

        # --- Layers ---
        # Embedding layer to convert token IDs to vectors
        self.embedding = nn.Embedding(self.vocab_size, config.embedding_dim)

        # GRU layer
        self.gru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,  # Input and output tensors are provided as (batch, seq, feature)
            dropout=config.dropout if config.num_layers > 1 else 0
        )

        # Fully connected layer to map GRU output to vocabulary size
        self.fc = nn.Linear(config.hidden_dim, self.vocab_size)

    def forward(self, x, h=None):
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length).
            h (torch.Tensor, optional): The initial hidden state. Defaults to None.

        Returns:
            torch.Tensor: The output logits of shape (batch_size, sequence_length, vocab_size).
            torch.Tensor: The final hidden state.
        """
        # 1. Embed the input sequence
        embedded = self.embedding(x)

        # 2. Pass the embedded sequence through the GRU
        # gru_out shape: (batch_size, sequence_length, hidden_dim)
        # h_n shape: (num_layers, batch_size, hidden_dim)
        gru_out, h_n = self.gru(embedded, h)

        # 3. Pass the GRU output through the fully connected layer
        # This gives us the logits for the next token prediction
        logits = self.fc(gru_out)

        return logits, h_n
