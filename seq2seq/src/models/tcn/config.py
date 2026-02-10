import torch

# TCN model configuration

# --- Data parameters ---
# vocab_size: The number of unique tokens in the dataset.
# (This will be set dynamically based on the dataset)
vocab_size = None

# --- Model architecture ---
# embedding_dim: The dimensionality of the token embeddings.
embedding_dim = 256

# num_channels: A list containing the number of channels in each convolutional layer.
# The length of this list determines the number of layers.
# Example for 4 layers: [256, 256, 256, 256]
num_channels = [256] * 4

# kernel_size: The size of the convolutional kernel.
kernel_size = 3

# dropout: The dropout probability for the TCN layers.
dropout = 0.2

# --- Training parameters ---
# learning_rate: The initial learning rate for the optimizer.
learning_rate = 1e-3

# batch_size: The number of sequences in each training batch.
batch_size = 64

# num_epochs: The total number of epochs to train for.
num_epochs = 10

# --- Hardware ---
# device: The device to run the model on ('cuda' or 'cpu').
device = 'cuda' if torch.cuda.is_available() else 'cpu'
