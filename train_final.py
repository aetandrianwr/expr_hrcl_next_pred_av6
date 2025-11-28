"""
Final training script with optimized model.
"""
import os
import random
import numpy as np
import torch

from src.configs import Config
from src.data import create_dataloaders
from src.models.optimized_transformer import OptimizedHierarchicalTransformer
from src.training.multitask_trainer import MultiTaskTrainer
from src.evaluation import Evaluator


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    config = Config()
    set_seed(config.SEED)
    
    print("="*70)
    print("GeoLife Next-Location Prediction - Final Optimized Model")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Device: {config.DEVICE}")
    print(f"  Model: Frequency-Aware Hierarchical Transformer")
    print(f"  D_MODEL: {config.D_MODEL}")
    print(f"  N_LAYERS: {config.N_LAYERS}")
    print(f"  N_HEADS: {config.N_HEADS}")
    print(f"  D_FF: {config.D_FF}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Val: {len(val_loader.dataset)}")
    print(f"  Test: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = OptimizedHierarchicalTransformer(config)
    num_params = model.count_parameters()
    print(f"  Parameters: {num_params:,}")
    print(f"  Budget: < 500,000")
    print(f"  Status: {'✓ PASS' if num_params < 500000 else '✗ FAIL'}")
    
    if num_params >= 500000:
        raise ValueError(f"Model exceeds 500K parameter limit!")
    
    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    trainer = MultiTaskTrainer(model, train_loader, val_loader, config)
    best_val_acc = trainer.train()
    
    # Evaluate
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    checkpoint = torch.load(os.path.join(config.CHECKPOINT_DIR, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    evaluator = Evaluator(model, test_loader, config)
    test_perf, _, _ = evaluator.evaluate()
    evaluator.print_results(test_perf)
    
    # Check target
    target_acc = 50.0
    print(f"\nTarget Acc@1: ≥{target_acc}%")
    print(f"Achieved Acc@1: {test_perf['acc@1']:.2f}%")
    if test_perf['acc@1'] >= target_acc:
        print("✓✓✓ TARGET ACHIEVED! ✓✓✓")
    else:
        gap = target_acc - test_perf['acc@1']
        print(f"✗ Gap to target: {gap:.2f}%")
    
    return test_perf


if __name__ == '__main__':
    main()
