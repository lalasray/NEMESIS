# NEMESIS

**Neural Memory-augmented Symbolic Interface for Sensors**

NEMESIS translates raw IMU sensor data into human activity labels through two complementary pathways:

1. **Translator Mode** — VQ-VAE tokens → Transformer → neuro-symbolic text → LLM classification, trained end-to-end with PPO reinforcement learning
2. **Descriptor Mode** — VQ-VAE tokens → statistical text description → LLM classification, zero-shot (no RL needed)

Both pathways use a VQ-VAE to learn a codebook of motion primitives from raw IMU signals, then leverage an LLM (OpenAI) to classify the activity.

## Architecture

### Translator Mode (RL-trained)

```
IMU Stream ──► VQ-VAE Tokenizer ──► Transformer ──► Neuro-Symbolic Text ──► LLM ──► Activity
                 (424K params)      (7.8M params)      (192 tokens)         │
                                         ▲                                   │
                                         └────────── PPO Reward ────────────┘
```

### Descriptor Mode (zero-shot)

```
IMU Stream ──► VQ-VAE Tokenizer ──► Token Statistics ──► LLM ──► Activity
                 (424K params)       (frequencies,         │
                                      entropy,             │
                                      transitions,         │
                                      bursts, ...)         │
                                                    (no training needed)
```

## Project Structure

```
NEMESIS/
├── nemesis/
│   ├── __init__.py            # Package init
│   ├── config.py              # Centralised hyperparameters and paths
│   ├── datasets.py            # Dataset loaders (UCI HAR, Opportunity)
│   ├── imu_tokenizer.py       # IMU → discrete tokens (Binning or VQ-VAE)
│   ├── neuro_symbolic.py      # Symbolic language spec, vocab, parser
│   ├── translator.py          # Transformer Encoder-Decoder + value head
│   ├── token_descriptor.py    # Token sequence → statistical text description
│   ├── rl_trainer.py          # PPO trainer + OpenAI reward function
│   ├── memory.py              # Persistent memory (ChromaDB + SQLite)
│   └── pipeline.py            # Full orchestrator (both modes)
├── train_har.py               # CLI training and evaluation script
├── requirements.txt           # Python dependencies
├── apikey.txt                 # OpenAI API key (not committed)
└── LICENSE                    # MIT License
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
echo "sk-your-key-here" > apikey.txt
```

### 3. Run Descriptor Mode (zero-shot, fastest to try)

```bash
python train_har.py --dataset opportunity --use-vqvae --descriptor-mode --workers 8
```

This will:
1. Download the Opportunity dataset (4 activities: Stand, Walk, Sit, Lie)
2. Pre-train VQ-VAE on raw IMU data (~200 epochs)
3. Generate statistical descriptions of token sequences
4. Classify activities via LLM — no Transformer or RL training needed

### 4. Run Translator Mode (RL-trained)

```bash
python train_har.py --dataset opportunity --use-vqvae --rl-epochs 30 --workers 8
```

This adds RL training of the Transformer to produce neuro-symbolic text.

### 5. Quick test (10% of data)

```bash
python train_har.py --dataset opportunity --use-vqvae --train-fraction 0.1 --workers 8
```

## CLI Reference

```
python train_har.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `uci_har` | Dataset: `uci_har`, `opportunity` |
| `--use-vqvae` | off | Use VQ-VAE tokenizer (recommended) |
| `--descriptor-mode` | off | Zero-shot: skip Translator/RL, use token statistics |
| `--rl-epochs` | 30 | Max RL training epochs |
| `--batch-size` | 16 | PPO batch size |
| `--lr` | 3e-4 | Learning rate |
| `--patience` | 4 | Early stopping patience (macro F1) |
| `--train-fraction` | 1.0 | Fraction of data per epoch (0.1 = 10%) |
| `--workers` | 8 | Parallel OpenAI API threads |
| `--eval-only` | off | Evaluate only (no training) |
| `--eval-samples` | 0 | Eval samples (0 = full test set) |
| `--reward-model` | `gpt-4.1-mini` | OpenAI model for classification |
| `--device` | `cpu` | PyTorch device |
| `--checkpoint` | none | Load checkpoint (e.g. `best`, `latest`) |
| `--skip-warmup` | off | Skip warm-start phase |
| `--vqvae-epochs` | 200 | VQ-VAE pre-training epochs |
| `--vqvae-patience` | 15 | VQ-VAE early stopping patience |
| `--retrain-vqvae` | off | Force re-train VQ-VAE |

## Components

### VQ-VAE Tokenizer (`imu_tokenizer.py`)

Learns a codebook of 1028 motion primitives from raw IMU signals via Vector Quantized Variational Autoencoder with EMA codebook updates.

- **Input**: `(T, 6)` raw IMU (accelerometer + gyroscope)
- **Output**: `T/4` discrete codebook indices
- **Architecture**: Conv1d encoder (stride 2×2 = 4× downsample) → VQ-EMA → Conv1d decoder
- **Parameters**: 424K

### Token Descriptor (`token_descriptor.py`)

Converts VQ-VAE token sequences into rich statistical text for direct LLM classification:

- Token frequencies and percentages
- Shannon entropy and codebook usage
- Temporal structure (start/middle/end breakdown)
- Bigram and trigram transition patterns
- Burst detection (repeated same-token runs)
- Self-repetition rate

### Translator (`translator.py`)

Transformer Encoder-Decoder (7.8M params) that maps VQ-VAE token sequences to neuro-symbolic text. Includes a value head for PPO.

### Neuro-Symbolic Language (`neuro_symbolic.py`)

192-token structured vocabulary describing motion patterns:

```
MOTION(limb=right_arm, type=swing, intensity=high)
GAIT(pattern=stride, frequency=2)
POSTURE(state=upright, transition=stable)
CONTEXT(duration=3, repetitions=4)
```

### RL Trainer (`rl_trainer.py`)

- **PPO** with clipped objective, entropy annealing, LR warmup + cosine decay
- OpenAI LLM classifies symbolic output → reward signal
- Auto-calibrated reward: `calibrate_rewards()` guarantees no single-class prediction strategy is profitable for any dataset
- Parallel API calls via `ThreadPoolExecutor` (~8× speedup)

### Datasets (`datasets.py`)

- **Opportunity**: 4 activities (Stand, Walk, Sit, Lie), BACK IMU 6ch @ 30Hz, variable-length segments, 4-subject split
- **UCI HAR**: 6 activities, waist IMU 9ch @ 50Hz, fixed 128-sample windows
- Class-balanced sampling and inverse-frequency class weights (capped)

### Memory (`memory.py`)

- **Vector Memory** (ChromaDB): similarity search over past translations
- **Knowledge Graph** (SQLite): cached pattern → activity rules
- Cross-session persistence

## Configuration

All hyperparameters are in `nemesis/config.py`:

| Config | Parameter | Default |
|--------|-----------|---------|
| IMU | `num_channels` | 6 |
| IMU | `codebook_size` | 1028 |
| IMU | `vq_embedding_dim` | 128 |
| Translator | `d_model` | 256 |
| Translator | `nhead` | 8 |
| Translator | `num_encoder_layers` | 4 |
| Translator | `max_seq_len` | 1024 |
| RL | `lr` | 3e-4 |
| RL | `clip_epsilon` | 0.2 |
| RL | `entropy_coef` | 0.08 (decays 0.95×/epoch, floor 0.02) |
| RL | `correct_reward` | 1.0 |
| RL | `wrong_reward` | auto-calibrated per dataset |

## License

MIT — see [LICENSE](LICENSE).
