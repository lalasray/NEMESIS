# NEMESIS

**Neural Embedded Motion Encoding with Statistical Inference for Sensors**

NEMESIS classifies human activities from raw IMU sensor data using a VQ-VAE to learn motion primitives and an LLM to classify their statistical descriptions — zero-shot, no supervised training needed.

## Architecture

```
IMU Stream ──► VQ-VAE Tokenizer ──► Token Statistics ──► LLM ──► Activity
                 (424K params)       (frequencies,
                                      entropy,
                                      transitions,
                                      bursts, ...)
```

1. **VQ-VAE** learns a codebook of 1028 motion primitives from raw IMU signals (unsupervised)
2. **Token Descriptor** computes statistical features of the codebook token sequence
3. **LLM** (OpenAI) classifies the activity from the statistical description

## Project Structure

```
NEMESIS/
├── nemesis/
│   ├── __init__.py            # Package init (v0.3.0)
│   ├── config.py              # Hyperparameters and paths
│   ├── datasets.py            # Dataset loaders (UCI HAR, Opportunity)
│   ├── imu_tokenizer.py       # IMU → discrete tokens (VQ-VAE)
│   ├── token_descriptor.py    # Token sequence → statistical text
│   ├── classifier.py          # OpenAI LLM classifier + reward calibration
│   └── pipeline.py            # Full orchestrator
├── train_har.py               # CLI evaluation script
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

### 3. Run

```bash
python train_har.py --dataset opportunity --workers 8
```

This will:
1. Download the Opportunity dataset (4 activities: Stand, Walk, Sit, Lie)
2. Pre-train VQ-VAE on raw IMU data (~200 epochs, unsupervised)
3. Tokenize test samples → compute statistical descriptions
4. Classify activities via LLM in parallel
5. Report macro F1, per-class precision/recall/F1

### Quick test (100 samples)

```bash
python train_har.py --dataset opportunity --eval-samples 100 --workers 8
```

## CLI Reference

```
python train_har.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `opportunity` | Dataset: `uci_har`, `opportunity` |
| `--eval-samples` | `0` | Eval samples (0 = full test set) |
| `--model` | `gpt-4.1-mini` | OpenAI model for classification |
| `--device` | `cpu` | PyTorch device |
| `--workers` | `8` | Parallel OpenAI API threads |
| `--vqvae-epochs` | `200` | VQ-VAE pre-training epochs |
| `--vqvae-patience` | `15` | VQ-VAE early stopping patience |
| `--retrain-vqvae` | off | Force re-train VQ-VAE |

## Components

### VQ-VAE Tokenizer (`imu_tokenizer.py`)

Learns a codebook of 1028 motion primitives from raw IMU via Vector Quantized VAE with EMA codebook updates.

- **Input**: `(T, 6)` raw IMU (accelerometer + gyroscope)
- **Output**: `T/4` discrete codebook indices
- **Architecture**: Conv1d encoder (stride 2×2 = 4× downsample) → VQ-EMA → Conv1d decoder
- **Parameters**: 424K

### Token Descriptor (`token_descriptor.py`)

Converts VQ-VAE token sequences into rich statistical text:

- Token frequencies and percentages
- Shannon entropy and codebook usage
- Temporal structure (start/middle/end breakdown)
- Bigram and trigram transition patterns
- Burst detection (repeated same-token runs)
- Self-repetition rate

### Classifier (`classifier.py`)

- OpenAI LLM classifies activity from token descriptor text
- Parallel API calls via `ThreadPoolExecutor` (~8× speedup)
- Auto-calibrated rewards: `calibrate_rewards()` ensures no single-class prediction strategy is profitable

### Datasets (`datasets.py`)

- **Opportunity**: 4 activities (Stand, Walk, Sit, Lie), BACK IMU 6ch @ 30Hz, variable-length, 4-subject split
- **UCI HAR**: 6 activities, waist IMU 9ch @ 50Hz, fixed 128-sample windows
- Class-balanced sampling and inverse-frequency class weights (capped at 3.0)

## Configuration

All hyperparameters in `nemesis/config.py`:

| Config | Parameter | Default |
|--------|-----------|---------|
| IMU | `num_channels` | 6 |
| IMU | `codebook_size` | 1028 |
| IMU | `vq_embedding_dim` | 128 |
| Classifier | `model` | `gpt-4.1-mini` |
| Classifier | `correct_reward` | 1.0 |
| Classifier | `wrong_reward` | auto-calibrated |

## License

MIT — see [LICENSE](LICENSE).
