import torch
import torch.nn as nn
from . import config

class VAE(nn.Module):
    """
    A Variational Autoencoder (VAE) with a GRU-based encoder and decoder.
    """
    def __init__(self, vocab_size):
        """
        Initializes the VAE model.

        Args:
            vocab_size (int): The number of unique tokens in the vocabulary.
        """
        super(VAE, self).__init__()
        self.vocab_size = vocab_size

        # --- Encoder ---
        self.encoder_embedding = nn.Embedding(self.vocab_size, config.embedding_dim)
        self.encoder_gru = nn.GRU(
            config.embedding_dim, config.hidden_dim, config.num_layers, 
            batch_first=True, dropout=config.dropout
        )
        self.fc_mu = nn.Linear(config.hidden_dim, config.z_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim, config.z_dim)

        # --- Decoder ---
        self.decoder_embedding = nn.Embedding(self.vocab_size, config.embedding_dim)
        self.decoder_gru = nn.GRU(
            config.embedding_dim + config.z_dim, config.hidden_dim, config.num_layers, 
            batch_first=True, dropout=config.dropout
        )
        self.decoder_fc = nn.Linear(config.hidden_dim, self.vocab_size)

    def encode(self, x):
        """
        Encodes the input sequence into a latent distribution (mu and logvar).

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: The mean (mu) of the latent distribution.
            torch.Tensor: The log variance (logvar) of the latent distribution.
        """
        embedded = self.encoder_embedding(x)
        _, hidden = self.encoder_gru(embedded)
        # Use the hidden state of the last layer
        hidden = hidden[-1]
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Performs the reparameterization trick to sample from the latent space.

        Args:
            mu (torch.Tensor): The mean of the latent distribution.
            logvar (torch.Tensor): The log variance of the latent distribution.

        Returns:
            torch.Tensor: A sampled latent vector (z).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x):
        """
        Decodes a latent vector into a sequence.

        Args:
            z (torch.Tensor): The latent vector of shape (batch_size, z_dim).
            x (torch.Tensor): The input tensor for teacher forcing.

        Returns:
            torch.Tensor: The output logits of shape (batch_size, sequence_length, vocab_size).
        """
        # Repeat z for each token in the sequence
        z = z.unsqueeze(1).repeat(1, x.size(1), 1)
        embedded = self.decoder_embedding(x)
        z_and_embedded = torch.cat([z, embedded], dim=2)
        output, _ = self.decoder_gru(z_and_embedded)
        logits = self.decoder_fc(output)
        return logits

    def forward(self, x):
        """
        Performs a forward pass through the VAE.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The reconstructed output logits.
            torch.Tensor: The mean of the latent distribution.
            torch.Tensor: The log variance of the latent distribution.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z, x)
        return recon_logits, mu, logvar

def vae_loss(recon_x, x, mu, logvar, kl_weight=1.0):
    """
    Calculates the VAE loss (Reconstruction + KL Divergence).

    Args:
        recon_x (torch.Tensor): The reconstructed output logits.
        x (torch.Tensor): The original input tensor.
        mu (torch.Tensor): The mean of the latent distribution.
        logvar (torch.Tensor): The log variance of the latent distribution.
        kl_weight (float, optional): The weight for the KL divergence term. Defaults to 1.0.

    Returns:
        torch.Tensor: The total VAE loss.
    """
    # Reconstruction loss (Cross-Entropy)
    recon_loss = nn.functional.cross_entropy(recon_x.view(-1, recon_x.size(-1)), x.view(-1), reduction='mean')

    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld / x.size(0) # Normalize by batch size

    return recon_loss + kl_weight * kld
