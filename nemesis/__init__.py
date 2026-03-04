"""
NEMESIS — Neural Memory-augmented Symbolic Interface for Sensors

Translates raw IMU sensor data into human activity labels through:
  1. VQ-VAE tokenization (learned motion primitives)
  2. Transformer translation to neuro-symbolic text (RL-trained), or
  3. Statistical token description (zero-shot descriptor mode)
  4. LLM classification via OpenAI API
"""

__version__ = "0.2.0"
