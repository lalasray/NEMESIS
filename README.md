# NEMESIS

**Neural Memory-augmented Symbolic Interface for Sensors**

Translates IMU sensor data into neuro-symbolic representations, interprets them via OpenAI, and continuously improves through reinforcement learning with persistent cross-session memory.

## Architecture

```
IMU Stream ──► [Tokenizer] ──► [Translator] ──► Neuro-Symbolic ──► [OpenAI] ──► Activity
                                     ▲                                  │
                                     └──── RL Reward ──────────────────┘
                                                                        │
                                                               [Persistent Memory]
```

## Project Structure

```
nemesis/
├── __init__.py          # Package init
├── config.py            # All hyperparameters and paths
├── imu_tokenizer.py     # IMU → discrete tokens (Binning or VQ-VAE)
├── neuro_symbolic.py    # Symbolic language spec, vocab, parser
├── translator.py        # Transformer Encoder-Decoder model
├── rl_trainer.py        # PPO trainer + OpenAI reward function
├── memory.py            # Persistent memory (ChromaDB + SQLite)
└── pipeline.py          # Full orchestrator
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

Place your key in `apikey.txt` (already done).

### 3. Run the demo

```bash
python -m nemesis.pipeline
```

This will:
1. Pretrain the Translator on synthetic IMU data
2. Translate 5 activities (walking, running, sitting, jumping, waving)
3. Run one RL update with OpenAI rewards
4. Test memory recall

### 4. Use in your own code

```python
from nemesis.pipeline import NemesisPipeline
from nemesis.imu_tokenizer import generate_synthetic_imu

# Initialize
pipeline = NemesisPipeline(device="cpu")
pipeline.start_session("my experiment")

# Pretrain (do once)
pipeline.supervised_pretrain(num_epochs=50)

# Translate real IMU data
import numpy as np
imu_data = np.load("my_imu_recording.npy")  # shape: (timesteps, 6)
result = pipeline.translate(imu_data, ground_truth="walking")

print(result.activity)      # "The person is walking at a moderate pace"
print(result.confidence)    # 0.85
print(result.symbolic_text) # "GAIT(pattern=stride, frequency=2)\n..."
print(result.from_cache)    # False (first time), True (subsequent)

# RL update happens automatically when buffer is full
pipeline.rl_train_step()

# End session — persists memory for next time
pipeline.end_session()
```

## Components

### IMU Tokenizer (`imu_tokenizer.py`)
Two strategies:
- **BinningTokenizer**: Fast, no training needed. Hashes statistical features of IMU windows.
- **VQVAE_Tokenizer**: Learns a codebook of motion primitives. Better quality, requires training.

### Translator (`translator.py`)
Transformer Encoder-Decoder (~4M params) that maps IMU token sequences to neuro-symbolic token sequences. Includes a value head for RL.

### Neuro-Symbolic Language (`neuro_symbolic.py`)
Structured intermediate representation:
```
MOTION(limb=right_arm, type=swing, intensity=high)
GAIT(pattern=stride, frequency=2)
POSTURE(state=upright, transition=stable)
CONTEXT(duration=3, repetitions=4)
```

### RL Trainer (`rl_trainer.py`)
- **PPO** with clipped objective
- OpenAI API provides reward signal (rates symbolic output quality)
- Supervised pretraining bootstraps before RL

### Memory (`memory.py`)
- **Vector Memory** (ChromaDB): similarity search over past translations
- **Knowledge Graph** (SQLite): cached pattern→activity rules
- Cross-session persistence — knowledge accumulates over time

## Configuration

All hyperparameters are in `nemesis/config.py`. Key settings:

| Config | Parameter | Default |
|--------|-----------|---------|
| IMU | `num_channels` | 6 (accel xyz + gyro xyz) |
| IMU | `window_size` | 25 samples |
| IMU | `codebook_size` | 512 tokens |
| Translator | `d_model` | 256 |
| Translator | `num_encoder_layers` | 4 |
| RL | `lr` | 1e-4 |
| RL | `clip_epsilon` | 0.2 |
| Memory | `confidence_threshold` | 0.85 |

## License

See [LICENSE](LICENSE).
