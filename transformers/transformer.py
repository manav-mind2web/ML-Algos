"""
Transformer Implementation from Scratch

This module provides a complete implementation of the Transformer architecture
as described in "Attention Is All You Need" (Vaswani et al., 2017).
"""

import numpy as np
import math


class MultiHeadAttention:
    """Multi-Head Attention mechanism."""

    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Initialize weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1

    def split_heads(self, x):
        """Split the last dimension into (num_heads, d_k)."""
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Calculate the attention weights."""
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores + (mask * -1e9)

        attention_weights = self.softmax(scores)
        return np.matmul(attention_weights, V), attention_weights

    def softmax(self, x):
        """Compute softmax values."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, query, key, value, mask=None):
        """Forward pass for multi-head attention."""
        batch_size = query.shape[0]

        Q = np.matmul(query, self.W_q)
        K = np.matmul(key, self.W_k)
        V = np.matmul(value, self.W_v)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)

        attention_output = attention_output.transpose(0, 2, 1, 3)
        concat_attention = attention_output.reshape(batch_size, -1, self.d_model)

        return np.matmul(concat_attention, self.W_o)


class FeedForward:
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def forward(self, x):
        """Forward pass."""
        hidden = self.relu(np.matmul(x, self.W1) + self.b1)
        return np.matmul(hidden, self.W2) + self.b2


class LayerNorm:
    """Layer Normalization."""

    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x):
        """Forward pass."""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class PositionalEncoding:
    """Sinusoidal Positional Encoding."""

    def __init__(self, d_model, max_seq_length):
        self.d_model = d_model
        self.encoding = self._create_encoding(max_seq_length, d_model)

    def _create_encoding(self, max_seq_length, d_model):
        """Create positional encoding matrix."""
        position = np.arange(max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        encoding = np.zeros((max_seq_length, d_model))
        encoding[:, 0::2] = np.sin(position * div_term)
        encoding[:, 1::2] = np.cos(position * div_term)

        return encoding

    def forward(self, x):
        """Add positional encoding to input."""
        seq_len = x.shape[1]
        return x + self.encoding[:seq_len]


class EncoderLayer:
    """Single Encoder Layer."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = dropout

    def forward(self, x, mask=None):
        """Forward pass."""
        attn_output = self.self_attention.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_output)

        ff_output = self.feed_forward.forward(x)
        x = self.norm2.forward(x + ff_output)

        return x


class DecoderLayer:
    """Single Decoder Layer."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = dropout

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """Forward pass."""
        self_attn_output = self.self_attention.forward(x, x, x, tgt_mask)
        x = self.norm1.forward(x + self_attn_output)

        cross_attn_output = self.cross_attention.forward(x, enc_output, enc_output, src_mask)
        x = self.norm2.forward(x + cross_attn_output)

        ff_output = self.feed_forward.forward(x)
        x = self.norm3.forward(x + ff_output)

        return x


class Transformer:
    """Complete Transformer model."""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 max_seq_length=100, dropout=0.1):
        self.d_model = d_model

        # Embeddings
        self.src_embedding = np.random.randn(src_vocab_size, d_model) * 0.1
        self.tgt_embedding = np.random.randn(tgt_vocab_size, d_model) * 0.1

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Encoder layers
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ]

        # Decoder layers
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ]

        # Final linear layer
        self.final_linear = np.random.randn(d_model, tgt_vocab_size) * 0.1

    def encode(self, src, src_mask=None):
        """Encode the source sequence."""
        x = self.src_embedding[src] * math.sqrt(self.d_model)
        x = self.positional_encoding.forward(x)

        for layer in self.encoder_layers:
            x = layer.forward(x, src_mask)

        return x

    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        """Decode the target sequence."""
        x = self.tgt_embedding[tgt] * math.sqrt(self.d_model)
        x = self.positional_encoding.forward(x)

        for layer in self.decoder_layers:
            x = layer.forward(x, enc_output, src_mask, tgt_mask)

        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """Forward pass through the transformer."""
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        output = np.matmul(dec_output, self.final_linear)
        return output

    @staticmethod
    def create_causal_mask(size):
        """Create a causal (look-ahead) mask for the decoder."""
        mask = np.triu(np.ones((size, size)), k=1)
        return mask

    @staticmethod
    def create_padding_mask(seq, pad_token=0):
        """Create a padding mask."""
        return (seq == pad_token).astype(np.float32)[:, np.newaxis, np.newaxis, :]


def demo():
    """Demonstrate transformer usage."""
    print("Transformer Demo")
    print("=" * 50)

    # Model parameters
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 64
    num_heads = 4
    num_encoder_layers = 2
    num_decoder_layers = 2
    d_ff = 256
    max_seq_length = 20

    # Create model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length
    )

    print(f"Model created with d_model={d_model}, {num_heads} heads")
    print(f"Encoder layers: {num_encoder_layers}, Decoder layers: {num_decoder_layers}")

    # Sample input
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8

    src = np.random.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = np.random.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    print(f"\nInput shapes:")
    print(f"  Source: {src.shape}")
    print(f"  Target: {tgt.shape}")

    # Create masks
    tgt_mask = Transformer.create_causal_mask(tgt_seq_len)

    # Forward pass
    output = model.forward(src, tgt, tgt_mask=tgt_mask)

    print(f"\nOutput shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {tgt_seq_len}, {tgt_vocab_size})")

    # Get predictions
    predictions = np.argmax(output, axis=-1)
    print(f"\nPredicted token indices (batch 0): {predictions[0]}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    demo()
