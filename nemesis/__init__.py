"""
NEMESIS — Neural Embedded Motion Encoding with Statistical Inference for Sensors

Classifies human activities from raw IMU sensor data through:
  1. VQ-VAE tokenization (learned motion primitives)
  2. Statistical token description (frequencies, entropy, transitions)
  3. Hierarchical memory retrieval (few-shot grounding)
  4. LLM classification via OpenAI API (few-shot)
"""

__version__ = "0.4.0"
