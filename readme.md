# ML-Algos

A collection of machine learning algorithm implementations from scratch using NumPy.

## Topics

### Transformers
Implementation of the Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017).

**Features:**
- Multi-Head Self-Attention mechanism
- Positional Encoding (sinusoidal)
- Layer Normalization
- Position-wise Feed-Forward Networks
- Complete Encoder-Decoder architecture
- Causal masking for autoregressive decoding

**Usage:**
```python
from transformers import Transformer

model = Transformer(
    num_layers=6,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    src_vocab_size=10000,
    tgt_vocab_size=10000
)

output = model.forward(src_tokens, tgt_tokens)
```

## Requirements
- Python 3.8+
- NumPy
