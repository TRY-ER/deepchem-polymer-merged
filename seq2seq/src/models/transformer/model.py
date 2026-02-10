
import torch
import torch.nn as nn
import math

from .config import MODEL_CONFIG


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer so it moves with the model to GPU/CPU
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Access the registered buffer correctly using getattr
        pe_buffer = getattr(self, 'pe')
        return x + pe_buffer[:x.size(0), :].to(x.device)


class TransformerModel(nn.Module):
    """
    Generative Transformer model for SMILES sequence generation.
    
    This model uses a decoder-only architecture suitable for autoregressive text generation.
    It outputs logits for each position in the vocabulary, enabling next-token prediction.
    """
    
    def __init__(self, vocab_size):
        super(TransformerModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = MODEL_CONFIG["embedding_dim"]
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model)
        
        # Transformer decoder layers (for autoregressive generation)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=MODEL_CONFIG["n_head"],
            dim_feedforward=MODEL_CONFIG["dim_feedforward"],
            dropout=MODEL_CONFIG["dropout"],
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=MODEL_CONFIG["n_layers"]
        )
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(self.d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz):
        """Generate causal mask to prevent looking at future tokens."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x, memory=None):
        """
        Forward pass for generative transformer.
        
        Args:
            x: Input token sequences of shape (batch_size, seq_len)
            memory: Memory from encoder (not used in decoder-only model, kept for compatibility)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        
        # Token embeddings with scaling
        embedded = self.embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        # Transpose for positional encoding (seq_len, batch_size, d_model)
        embedded = embedded.transpose(0, 1)
        embedded = self.pos_encoding(embedded)
        embedded = embedded.transpose(0, 1)  # Back to (batch_size, seq_len, d_model)
        
        # Generate causal mask for autoregressive training
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        
        # For decoder-only architecture, we use the same sequence as both target and memory
        # This creates a causal transformer suitable for language modeling
        if memory is None:
            memory = embedded
        
        # Transformer decoder forward pass
        transformer_out = self.transformer_decoder(
            tgt=embedded,
            memory=memory,
            tgt_mask=tgt_mask
        )
        
        # Project to vocabulary space
        logits = self.output_projection(transformer_out)
        
        return logits
