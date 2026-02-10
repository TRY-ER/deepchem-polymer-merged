import torch

# Mamba model configuration

# --- Data parameters ---
# vocab_size: The number of unique tokens in the dataset.
# (This will be set dynamically based on the dataset)
vocab_size = None

# --- Model architecture ---
# d_model: The main dimension of the model.
d_model = 256

# n_layer: The number of Mamba blocks.
n_layer = 4

# d_state: The dimension of the state space model (SSM) state.
d_state = 16

# d_conv: The dimension of the 1D convolutional kernel.
d_conv = 4

# expand: The expansion factor for the feed-forward network within the Mamba block.
expand = 2

# --- Training parameters ---
# learning_rate: The initial learning rate for the optimizer.
learning_rate = 1e-3

# batch_size: The number of sequences in each training batch.
batch_size = 64

# num_epochs: The total number of epochs to train for.
num_epochs = 15

# --- Hardware ---
# device: The device to run the model on ('cuda' or 'cpu').
device = 'cuda' if torch.cuda.is_available() else 'cpu'
