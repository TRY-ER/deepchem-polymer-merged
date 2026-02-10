import torch
import torch.nn as nn
try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    # Fallback for older PyTorch versions
    from torch.nn.utils import weight_norm
from . import config


class CausalConv1d(nn.Module):
    """A 1D causal convolution layer.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the convolutional kernel.
        dilation (int, optional): The dilation rate. Defaults to 1.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        # Remove padding from kwargs if it exists to avoid conflict
        kwargs.pop('padding', None)
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        # Trim the padding to make it causal
        return x[:, :, :-self.padding] if self.padding > 0 else x


class TCNBlock(nn.Module):
    """A single block of a Temporal Convolutional Network.

    Args:
        n_inputs (int): The number of input channels.
        n_outputs (int): The number of output channels.
        kernel_size (int): The size of the convolutional kernel.
        stride (int): The stride of the convolution.
        dilation (int): The dilation rate.
        dropout (float, optional): The dropout rate. Defaults to 0.2.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        # Create causal convolutions and apply weight norm to the internal conv layers
        self.conv1_causal = CausalConv1d(n_inputs, n_outputs, kernel_size, stride=stride, dilation=dilation)
        self.conv1 = weight_norm(self.conv1_causal.conv)  # Apply weight norm to the internal conv layer
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2_causal = CausalConv1d(n_outputs, n_outputs, kernel_size, stride=stride, dilation=dilation)
        self.conv2 = weight_norm(self.conv2_causal.conv)  # Apply weight norm to the internal conv layer
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # No need for Sequential with lambda functions - handle in forward()
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # First convolution block (using the causal wrapper which has the weight-normed conv inside)
        out = self.conv1_causal(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second convolution block
        out = self.conv2_causal(out)  # Fix: use out, not x
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """A Temporal Convolutional Network (TCN) for sequence modeling.

    Args:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimension of the embeddings.
        num_channels (list): A list of the number of channels for each layer.
        kernel_size (int, optional): The size of the convolutional kernel. Defaults to 2.
        dropout (float, optional): The dropout rate. Defaults to 0.2.
    """
    def __init__(self, vocab_size, embedding_dim, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.tcn = self._create_tcn(embedding_dim, num_channels, kernel_size, dropout)
        self.decoder = nn.Linear(num_channels[-1], vocab_size)

    def _create_tcn(self, n_inputs, num_channels, kernel_size, dropout):
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = n_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embedding_dim)
        tcn_out = self.tcn(embedded.transpose(1, 2)).transpose(1, 2)
        # tcn_out shape: (batch_size, seq_len, num_channels[-1])
        output = self.decoder(tcn_out)
        # output shape: (batch_size, seq_len, vocab_size)
        return output, None # Returning None for compatibility with the GRU/LSTM structure
