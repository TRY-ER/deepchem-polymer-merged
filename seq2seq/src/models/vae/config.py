import torch

# VAE model configuration

# --- Data parameters ---
# vocab_size: The number of unique tokens in the dataset.
# (This will be set dynamically based on the dataset)
vocab_size = None

# --- Model architecture ---
# embedding_dim: The dimensionality of the token embeddings.
embedding_dim = 256

# hidden_dim: The number of features in the hidden state of the GRU encoder/decoder.
hidden_dim = 512

# z_dim: The dimensionality of the latent space vector (z).
z_dim = 128

# num_layers: The number of recurrent layers in the GRU encoder/decoder.
num_layers = 2

# dropout: The dropout probability for the GRU layers.
dropout = 0.2

# --- Training parameters ---
# learning_rate: The initial learning rate for the optimizer.
learning_rate = 1e-3

# batch_size: The number of sequences in each training batch.
batch_size = 64

# num_epochs: The total number of epochs to train for.
num_epochs = 20

# kl_weight: The weight for the Kullback-Leibler divergence loss term.
# This can be annealed (gradually increased) during training.
kl_weight = 0.0

# --- Hardware ---
# device: The device to run the model on ('cuda' or 'cpu').
device = 'cuda' if torch.cuda.is_available() else 'cpu'
