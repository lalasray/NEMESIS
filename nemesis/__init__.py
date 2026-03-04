"""
NEMESIS — Neural Embedded Motion Encoding with Statistical Inference for Sensors

Classifies human activities from raw IMU sensor data through:
  1. VQ-VAE tokenization (learned motion primitives)
  2. Statistical token description (frequencies, entropy, transitions)
  3. LLM classification via OpenAI API (zero-shot)
"""

__version__ = "0.3.0"
