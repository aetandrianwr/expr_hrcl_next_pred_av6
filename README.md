# GeoLife Next-Location Prediction

A hierarchical Transformer-based system for next-location prediction on the GeoLife dataset.

## Project Structure

```
.
├── src/
│   ├── configs/        # Configuration files
│   ├── data/           # Dataset and dataloaders
│   ├── models/         # Model architectures
│   ├── training/       # Training logic
│   ├── evaluation/     # Evaluation logic
│   └── utils/          # Metrics and utilities
├── data/               # GeoLife dataset
├── checkpoints/        # Saved model checkpoints
├── logs/               # Training logs
├── train.py            # Main training script
└── test.py             # Test evaluation script
```

## Model Architecture

Hierarchical Transformer with:
- Multi-resolution spatial encoding (H3 hierarchies)
- Temporal encoding (time-of-day, day-of-week)
- User embeddings
- Duration and time-gap features
- Multi-head self-attention
- < 500K parameters

## Usage

### Training
```bash
python train.py
```

### Evaluation
```bash
python test.py
```

## Requirements
- PyTorch
- NumPy
- scikit-learn
- tqdm
