"""
Transformer Architecture Implementation

This module implements the Transformer architecture as described in
"Attention Is All You Need" (Vaswani et al., 2017).
"""

import numpy as np
from typing import Optional, Tuple


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax values along the specified axis."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit activation function."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class LayerNorm:
    """Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MultiHeadAttention:
    """Multi-Head Self-Attention mechanism."""

    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Initialize weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split the last dimension into (num_heads, d_k)."""
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq, d_k)

    def scaled_dot_product_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute scaled dot-product attention."""
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        attention_weights = softmax(scores, axis=-1)
        output = np.matmul(attention_weights, V)

        return output, attention_weights

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass of multi-head attention."""
        batch_size, seq_len, _ = x.shape

        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        attention_output = attention_output.transpose(0, 2, 1, 3)
        attention_output = attention_output.reshape(batch_size, seq_len, self.d_model)

        output = np.matmul(attention_output, self.W_o)

        return output, attention_weights


class FeedForward:
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model: int, d_ff: int):
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with GELU activation."""
        hidden = gelu(np.matmul(x, self.W1) + self.b1)
        output = np.matmul(hidden, self.W2) + self.b2
        return output


class PositionalEncoding:
    """Sinusoidal Positional Encoding."""

    def __init__(self, d_model: int, max_seq_len: int = 5000):
        self.d_model = d_model
        self.encoding = self._create_encoding(max_seq_len, d_model)

    def _create_encoding(self, max_seq_len: int, d_model: int) -> np.ndarray:
        """Create positional encoding matrix."""
        position = np.arange(max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        encoding = np.zeros((max_seq_len, d_model))
        encoding[:, 0::2] = np.sin(position * div_term)
        encoding[:, 1::2] = np.cos(position * div_term)

        return encoding

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Add positional encoding to input embeddings."""
        seq_len = x.shape[1]
        return x + self.encoding[:seq_len]


class TransformerEncoderLayer:
    """Single Transformer Encoder Layer."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forward pass with residual connections and layer normalization."""
        # Self-attention with residual connection
        attention_output, _ = self.attention.forward(x, mask)
        x = self.norm1.forward(x + attention_output)

        # Feed-forward with residual connection
        ff_output = self.feed_forward.forward(x)
        x = self.norm2.forward(x + ff_output)

        return x


class TransformerEncoder:
    """Transformer Encoder stack."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        max_seq_len: int = 5000
    ):
        self.d_model = d_model

        # Embedding layer (simplified as random initialization)
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        self.layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forward pass through the encoder stack."""
        # Get embeddings
        x = self.embedding[x] * np.sqrt(self.d_model)
        x = self.positional_encoding.forward(x)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer.forward(x, mask)

        return x


class TransformerDecoderLayer:
    """Single Transformer Decoder Layer."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(
        self,
        x: np.ndarray,
        encoder_output: np.ndarray,
        self_mask: Optional[np.ndarray] = None,
        cross_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forward pass with masked self-attention and cross-attention."""
        # Masked self-attention
        self_attn_output, _ = self.self_attention.forward(x, self_mask)
        x = self.norm1.forward(x + self_attn_output)

        # Cross-attention with encoder output
        cross_attn_output, _ = self.cross_attention.forward(x, cross_mask)
        x = self.norm2.forward(x + cross_attn_output)

        # Feed-forward
        ff_output = self.feed_forward.forward(x)
        x = self.norm3.forward(x + ff_output)

        return x


class Transformer:
    """Complete Transformer model for sequence-to-sequence tasks."""

    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        src_vocab_size: int = 10000,
        tgt_vocab_size: int = 10000,
        max_seq_len: int = 5000
    ):
        self.encoder = TransformerEncoder(
            num_layers, d_model, num_heads, d_ff, src_vocab_size, max_seq_len
        )

        self.d_model = d_model
        self.tgt_embedding = np.random.randn(tgt_vocab_size, d_model) * 0.02
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        self.decoder_layers = [
            TransformerDecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]

        # Output projection
        self.output_projection = np.random.randn(d_model, tgt_vocab_size) * 0.02

    def create_causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal mask for decoder self-attention."""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return (mask == 0).astype(np.float32)

    def forward(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
        src_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forward pass through the complete Transformer."""
        # Encode source sequence
        encoder_output = self.encoder.forward(src, src_mask)

        # Prepare target embeddings
        tgt_seq_len = tgt.shape[1]
        x = self.tgt_embedding[tgt] * np.sqrt(self.d_model)
        x = self.positional_encoding.forward(x)

        # Create causal mask for decoder
        causal_mask = self.create_causal_mask(tgt_seq_len)

        # Decode
        for layer in self.decoder_layers:
            x = layer.forward(x, encoder_output, causal_mask, src_mask)

        # Project to vocabulary
        logits = np.matmul(x, self.output_projection)

        return logits


def example_usage():
    """Demonstrate Transformer usage."""
    # Model configuration
    num_layers = 2
    d_model = 64
    num_heads = 4
    d_ff = 256
    vocab_size = 1000

    # Create model
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        max_seq_len=100
    )

    # Example input (batch_size=2, seq_len=10)
    src = np.random.randint(0, vocab_size, size=(2, 10))
    tgt = np.random.randint(0, vocab_size, size=(2, 8))

    # Forward pass
    output = transformer.forward(src, tgt)
    print(f"Input source shape: {src.shape}")
    print(f"Input target shape: {tgt.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output represents logits over vocabulary of size {vocab_size}")

    return transformer


if __name__ == "__main__":
    example_usage()
