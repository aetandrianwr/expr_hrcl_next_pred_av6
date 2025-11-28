"""
Test evaluation script - loads best model and evaluates on test set.
"""
import os
import torch

from src.configs import Config
from src.data import create_dataloaders
from src.models import HierarchicalTransformer
from src.evaluation import Evaluator


def main():
    config = Config()
    
    print("="*70)
    print("GeoLife Next-Location Prediction - Test Evaluation")
    print("="*70)
    
    # Create dataloaders
    print("\nLoading data...")
    _, _, test_loader = create_dataloaders(config)
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = HierarchicalTransformer(config)
    print(f"  Total parameters: {model.count_parameters():,}")
    
    # Load best model
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pt')
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Best validation Acc@1: {checkpoint['best_val_acc']:.2f}%")
    
    # Evaluate on test set
    evaluator = Evaluator(model, test_loader, config)
    test_perf, _, _ = evaluator.evaluate()
    evaluator.print_results(test_perf)
    
    # Check target
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
