"""
Main training script for GeoLife next-location prediction.
"""
import os
import random
import numpy as np
import torch

from src.configs import Config
from src.data import create_dataloaders
from src.models import HierarchicalTransformer
from src.training import Trainer
from src.evaluation import Evaluator


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Set seed
    config = Config()
    set_seed(config.SEED)
    
    print("="*70)
    print("GeoLife Next-Location Prediction with Hierarchical Transformer")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Device: {config.DEVICE}")
    print(f"  Seed: {config.SEED}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Model dimension: {config.D_MODEL}")
    print(f"  Number of layers: {config.N_LAYERS}")
    print(f"  Number of heads: {config.N_HEADS}")
    print(f"  Dropout: {config.DROPOUT}")
    print(f"  Max epochs: {config.NUM_EPOCHS}")
    print(f"  Patience: {config.PATIENCE}")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = HierarchicalTransformer(config)
    num_params = model.count_parameters()
    print(f"  Total parameters: {num_params:,}")
    print(f"  Parameter budget: < 500,000")
    print(f"  Status: {'✓ PASS' if num_params < 500000 else '✗ FAIL'}")
    
    if num_params >= 500000:
        raise ValueError(f"Model has {num_params:,} parameters, exceeding the 500K limit!")
    
    # Train model
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    trainer = Trainer(model, train_loader, val_loader, config)
    best_val_acc = trainer.train()
    
    # Load best model and evaluate on test set
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    print("\nLoading best model...")
    checkpoint = torch.load(os.path.join(config.CHECKPOINT_DIR, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    evaluator = Evaluator(model, test_loader, config)
    test_perf, _, _ = evaluator.evaluate()
    evaluator.print_results(test_perf)
    
    # Check if we met the target
    target_acc = 50.0
    print(f"\nTarget Acc@1: ≥{target_acc}%")
    print(f"Achieved Acc@1: {test_perf['acc@1']:.2f}%")
    if test_perf['acc@1'] >= target_acc:
        print("✓ TARGET ACHIEVED!")
    else:
        print(f"✗ Target not met (gap: {target_acc - test_perf['acc@1']:.2f}%)")
    
    return test_perf


if __name__ == '__main__':
    main()
