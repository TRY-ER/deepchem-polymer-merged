
import torch
import torch.nn as nn

from .config import MODEL_CONFIG


class LSTMModel(nn.Module):
    """
    A Long Short-Term Memory (LSTM) model for sequence generation.
    This is a generative model that outputs logits for each position in the vocabulary.
    """
    def __init__(self, vocab_size):
        """
        Initializes the LSTM model.

        Args:
            vocab_size (int): The number of unique tokens in the vocabulary.
        """
        super(LSTMModel, self).__init__()
        self.vocab_size = vocab_size

        # --- Layers ---
        # Embedding layer to convert token IDs to vectors
        self.embedding = nn.Embedding(self.vocab_size, MODEL_CONFIG["embedding_dim"])

        # LSTM layer (remove bidirectional for generative model)
        self.lstm = nn.LSTM(
            input_size=MODEL_CONFIG["embedding_dim"],
            hidden_size=MODEL_CONFIG["hidden_dim"],
            num_layers=MODEL_CONFIG["n_layers"],
            batch_first=True,  # Input and output tensors are provided as (batch, seq, feature)
            dropout=MODEL_CONFIG["dropout"] if MODEL_CONFIG["n_layers"] > 1 else 0,
            bidirectional=False  # Generative models should not be bidirectional
        )

        # Fully connected layer to map LSTM output to vocabulary size
        self.fc = nn.Linear(MODEL_CONFIG["hidden_dim"], self.vocab_size)

    def forward(self, x, h=None):
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length).
            h (tuple, optional): The initial hidden state (h_0, c_0). Defaults to None.

        Returns:
            torch.Tensor: The output logits of shape (batch_size, sequence_length, vocab_size).
            tuple: The final hidden state (h_n, c_n).
        """
        # 1. Embed the input sequence
        embedded = self.embedding(x)

        # 2. Pass the embedded sequence through the LSTM
        # lstm_out shape: (batch_size, sequence_length, hidden_dim)
        # (h_n, c_n) shape: (num_layers, batch_size, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(embedded, h)

        # 3. Pass the LSTM output through the fully connected layer
        # This gives us the logits for the next token prediction
        logits = self.fc(lstm_out)

        return logits, (h_n, c_n)
