import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, steps=10000):
        super().__init__()
        self.dim = dim
        self.steps = steps
        # due to pairs for sin and cos the dimension is doubled
        half = dim // 2
        # this function creates the frequency table for the sinusoidal time embedding
        freq = torch.exp(
            -math.log(steps) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freq", freq)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        # the tensor must be of shape [token length] other shapes like batch size are not supported here
        # The objective is to convert the token sequence into a time embedding
        t = t.float()
        # the dot product of the time and the frequency table resulting in a 2D tensor with length
        # [..t.shape[:-1], dim // 2]
        self.freq = self.freq.to(t.device)
        args = t[:, None] * self.freq[None]
        # to match the shape again the half is processed with sin and other half with cos and concatenated
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        # the mlp is applied to learn the feature at the end
        return self.mlp(emb)


class AdaLN(nn.Module):
    """
    Adaptive Layer Norm
    """

    def __init__(self, dim, cond_dim):
        super().__init__()
        self.cond_dim = cond_dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        # the projection layer learns from conditional matrix as input
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))
        # the initial weight and bias are set to zero as it adapts from condition and defines the initial scale and shift
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, cond):
        # not yet sure what should be the shape of input [1, size] or just [size]
        # if [size] it returns [size, size]
        # if [1, size] it returns [1, 1, size]
        # Need to see exactly where it is used to determine the functionality
        # initial scale and shift is derived from projection layer (updated during backprop)
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        x = self.norm(x)
        # normalization layer is imposed with scale and shift to have a conditional influence
        x = x * (1 + scale[:, None]) + shift[:, None]
        return x


class SwiGLU(nn.Module):
    """
    Swish Gated Linear Unit
    """

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return x


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) — relative positions,
    better for variable-length sequences.
    """

    def __init__(self, dim: int, max_seq_len: int = 1024):
        super().__init__()
        # the dim here is the sequence length of each attention head
        # the following part creates the frequency table for the rotary embedding
        # which takes head dimension and creates pairs to result in a 2D tensor with length
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        # to map the frequency table to the maximum sequence length
        t = torch.arange(max_seq_len).float()
        # this frequency values is imposed dot product with vector with progression to index-length to
        # increase the frequency value a the magnitude of the index
        # this will ensure the distance between farther token will be high and low for nearer tokens
        # it returns the tensor with shape (max_seq_len, dim // 2)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # the vector is of shape (max_seq_len, dim // 2) will not map directly to the embedding of shape (max_seq_len, dim)
        # so we have to concatenate the vector with itself to get the shape (max_seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        # register the buffer for cosine and sine values for the frequency table with initial 2 dimensions to be 1, 1
        # as the input q and k are of shape ( B, num_heads, L, seq_split_per_head )
        # for product it should be of shape ( 1, 1, L, seq_split_per_head )
        # but at init it is at shape ( 1, 1, max_seq_len, seq_split_per_head )
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        # the initialization of the cos and sin were of shape (1, 1, max_seq_len, seq_split_per_head)
        # the max_seq_len has to be limited to the q and k dimension to ensure the shape is correct for product
        seq_len = q.shape[2]
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        return self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, x, cos, sin):
        return x * cos + self._rotate_half(x) * sin


class MultiHeadSelfAttention(nn.Module):
    """Bidirectional multi-head self-attention with RoPE."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # the number of dim has to be divisible by n_heads
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        # dimension of each head
        self.head_dim = dim // n_heads
        # the scale is the denominator of the attention score that ensures the attention scores are not too large or too small
        # which is just the 1 / root of the head dimension
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, L, dim), mask: (B, L) bool (True = valid token)
        """
        B, L, _ = (
            x.shape
        )  # here B -> Batch size, L -> No. of vectors in batch, _ -> Number of tokens in each vector
        qkv = self.qkv(
            x
        )  # this increases the number of tokens in each vector to 3x (not the same content at each q, k, v)
        # the 3x dimension is split into 3 different batches with split for each heads
        # For example, if we provide input of 64 no. of tokens,
        # qkv -> 64 X 3 = 192 of length
        # qkv refactor shape -> 3 x 4 x 16 ; 3 -> split for (Q, K, V), 4 -> number of heads, 16 -> number of token for each head
        qkv = qkv.reshape(B, L, 3, self.n_heads, self.head_dim)
        # the permute alters the shape of qkv vector based on the indices in qkv.shape
        # here the (B, L, 3, num_heads, seq_split_per_head ) -> (3, B, num_heads, L, seq_split_per_head)
        # the unbind splits the qkv vector into x different vectors (where x is the index of the dimension in shape)
        # as here the index is 0 which ix "3" as value and it splits the vector into 3 different vectors -> q, k, v
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(
            0
        )  # each ( B, num_heads, L, seq_split_per_head )

        # implementing rotary positional embedding on key  and query
        q, k = self.rope(q, k)
        # attention mechanism is achieved by (q X k ) X scale ; here scale is 1 / sqrt(head_dim)
        # @ refers to matrix multiplication
        # the k transpose results the dimensiton to be (B, num_heads, seq_split_per_head, L)
        # while q is in dim (B, num_heads, L, seq_split_per_head)
        # resulting the final attention shape to be (B, num_heads, L, L) as L is higher than seq_split_per_head
        attn = (
            q @ k.transpose(-2, -1)
        ) * self.scale  # ( B, num_heads, L, seq_split_per_head )

        # applying mask
        if mask is not None:
            # mask: (B, L), True = valid → False means padding
            attn_mask = mask[:, None, None, :].float()
            attn = attn + (1 - attn_mask) * (-1e9)

        # applying softmax on the attention
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        # the v vector is in shape (B, num_heads, L, seq_split_per_head)
        # with matmul with attn the result becomes of shape (B, num_heads, L, seq_split_per_head)
        # to match the shape of the output to the input vector first transpose over (1, 2) is done
        # which causes (B, num_heads, L, seq_split_per_head) -> (B, L, num_heads, seq_split_per_head)
        # then the reshape is done for (B, L, num_heads * seq_split_per_head) to return the output shape merging the heads
        # resulting as (B, L, dim)
        out = (attn @ v).transpose(1, 2).reshape(B, L, self.dim)
        # final projection layer is implemented linearly
        return self.proj(out)


class DenoisingBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        # adaptive layer norm with condition dimension
        self.adln1 = AdaLN(dim, cond_dim)
        # multihead attention layer
        self.attn = MultiHeadSelfAttention(dim, n_heads, dropout)
        # adaptive layer norm with condition dimension
        self.adln2 = AdaLN(dim, cond_dim)
        # feedforward network with SwiGLU activation
        self.ffn = SwiGLU(dim)
        # dropout layer
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cond, mask):
        # self attention layer adaptive layer norm
        x = x + self.drop(self.attn(self.adln1(x, cond)))
        # feed forward layer with adaptive layer norm
        x = x + self.drop(self.ffn(self.adln2(x, cond)))
        return x


class PolyDiffusionTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len: int = 700,
        dim: int = 512,
        depth: int = 8,
        n_heads: int = 4,
        dropout: float = 0.1,
        cond_dim=None,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size

        # setting up sinusoidal time embedding
        self.time_emb = SinusoidalTimeEmbedding(dim)
        cond_total = dim
        self.use_cond = cond_dim is not None
        if self.use_cond:
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            cond_total = dim * 2

        # setting up token embedding (based on indexing with vocab_size)
        self.tok_emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        # setting up position embedding (based on maximum lenght of input / generation)
        # we need to analyze the dataset to determine the maximum length of input
        self.pos_emb = nn.Embedding(max_len, dim)
        # stacking of denoising blocks
        self.block = nn.ModuleList(
            [DenoisingBlock(dim, n_heads, cond_total, dropout) for _ in range(depth)]
        )
        # final layer normalization
        self.final_norm = nn.LayerNorm(dim)
        # output projection to match the vacab_size (this will give prediction of each token in vocab size)
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)
        # initializing weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, xt, t, mask, cond):
        # the xt would be in dimension (batch_size, sequence_length)
        # the xt would contain only integers to represent token ids from the tokenizer
        # the t would contain time steps as integers but not necessarily sequentially
        # extracting batch size and length in each batch
        B, L = xt.shape[:2]
        t = t.to(xt.device)
        # creating time embedding
        t_emb = self.time_emb(t)
        if self.use_cond and cond is not None:
            c_emb = self.cond_proj(cond)
            c_signal = torch.cat([t_emb, c_emb], dim=-1)
        else:
            c_signal = t_emb
            if self.use_cond:
                c_signal = torch.cat([t_emb, torch.zeros_like(c_signal)], dim=-1)

        # creating token embedding
        x = self.tok_emb(xt)

        # creating position embedding
        pos = torch.arange(L, device=xt.device).unsqueeze(0)

        # mat add token_embedding + position_embedding
        x = x + self.pos_emb(pos)

        # time embedding is passed to the denoising blocks along with the input tokens
        for block in self.block:
            x = block(x, c_signal, mask)

        # adding final normalization
        x = self.final_norm(x)
        # adding output projection (linear layer to match vacabulary size)
        return self.out_proj(x)
        # the final output layer will have a dimension of (time_steps, L, vocab_size)
        # it means it will predict for each time step for each component index for L (input sequence length)
        # of probability values for each token in the vocabulary.
        # means we can extract the highest probability for each position for every time step


if __name__ == "__main__":
    # # running the model for testing
    # model = MultiHeadSelfAttention(64, 4)
    # x = torch.randn(1, 32, 64)
    # mask = torch.ones(1, 32).bool()
    # out = model(x, mask)
    # print(out.shape)

    # # testing sinusoidal embedding
    # model = SinusoidalTimeEmbedding(32)
    # x = torch.randn(32)
    # out = model(x)
    # print(out.shape)

    # # testing adaptive layer norm
    # model = AdaLN(32, cond_dim=32)
    # cond = torch.randn(1, 32)
    # x = torch.randn(1, 32)
    # out = model(x, cond)
    # print(out.shape)

    # testing swish gated linear unit
    # model = SwiGLU(32)
    # x = torch.randn(1, 32)
    # out = model(x)
    # print(out.shape)

    # # testing denoising block
    # model = DenoisingBlock(32, 4, 32)
    # x = torch.randn(1, 32, 32)
    # cond = torch.randn(1, 32)
    # mask = torch.ones(1, 32).bool()
    # out = model(x, cond, mask)
    # print(out.shape)

    # # testing poly diffusion model
    model = PolyDiffusionTransformer(64)
    x = torch.randint(1, 32, (1, 32))
    t = torch.arange(14)
    cond = torch.randn(1, 32)
    mask = torch.ones(1, 32).bool()
    out = model(x, t, cond, mask)
    print(out)
    print(out.shape)
