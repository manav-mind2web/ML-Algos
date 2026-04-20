# Transformers

Transformers are a neural network architecture introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017). They have become the foundation for modern NLP models like BERT, GPT, and T5.

## Key Concepts

### 1. Self-Attention Mechanism
The core innovation of transformers. It allows the model to weigh the importance of different parts of the input when processing each element.

**Attention Formula:**
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

Where:
- Q (Query): What we're looking for
- K (Key): What we have to offer
- V (Value): The actual content
- d_k: Dimension of the key vectors (for scaling)

### 2. Multi-Head Attention
Instead of performing a single attention function, transformers use multiple attention heads in parallel. This allows the model to attend to information from different representation subspaces.

### 3. Positional Encoding
Since transformers don't have recurrence or convolution, they need a way to understand the order of the sequence. Positional encodings are added to the input embeddings to provide position information.

**Sinusoidal Positional Encoding:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 4. Feed-Forward Networks
Each transformer layer includes a position-wise feed-forward network applied to each position separately and identically.

### 5. Layer Normalization & Residual Connections
Each sub-layer has a residual connection followed by layer normalization:
```
output = LayerNorm(x + Sublayer(x))
```

## Architecture Overview

```
Input Embeddings + Positional Encoding
           |
    +------v------+
    |   Encoder   |  (N layers)
    |  - Multi-Head Self-Attention
    |  - Feed Forward Network
    +------+------+
           |
    +------v------+
    |   Decoder   |  (N layers)
    |  - Masked Multi-Head Self-Attention
    |  - Multi-Head Cross-Attention
    |  - Feed Forward Network
    +------+------+
           |
    Linear + Softmax
           |
        Output
```

## Applications

- **Natural Language Processing**: Machine translation, text generation, question answering
- **Computer Vision**: Vision Transformers (ViT), image classification
- **Speech**: Speech recognition, text-to-speech
- **Multimodal**: CLIP, DALL-E, combining text and images

## Files in This Directory

- `transformer.py` - Complete implementation of a transformer from scratch
- `README.md` - This documentation file

## Usage

```python
from transformer import Transformer

# Create a transformer model
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    max_seq_length=100,
    dropout=0.1
)

# Forward pass
output = model(src, tgt, src_mask, tgt_mask)
```

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need"
2. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"
3. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners" (GPT-2)
