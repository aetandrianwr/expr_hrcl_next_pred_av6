"""
Evaluation module for next-location prediction.
"""
import numpy as np
import torch
from tqdm import tqdm

from src.utils.metrics import calculate_correct_total_prediction, get_performance_dict


class Evaluator:
    """Evaluator class for the model."""
    
    def __init__(self, model, test_loader, config):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = config.DEVICE
        
        # Move model to device
        self.model.to(self.device)
    
    def evaluate(self):
        """Evaluate the model on test set."""
        self.model.eval()
        
        all_results = []
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for features, targets in tqdm(self.test_loader, desc='Testing'):
                # Move to device
                features = {k: v.to(self.device) for k, v in features.items()}
                targets = targets.to(self.device)
                
                # Forward pass
                logits = self.model(features)
                
                # Calculate metrics
                results, tgt_cpu, pred_cpu = calculate_correct_total_prediction(logits, targets)
                all_results.append(results)
                all_targets.append(tgt_cpu)
                all_predictions.append(pred_cpu)
        
        # Aggregate results
        all_results = np.sum(all_results, axis=0)
        return_dict = {
            "correct@1": all_results[0],
            "correct@3": all_results[1],
            "correct@5": all_results[2],
            "correct@10": all_results[3],
            "rr": all_results[4],
            "ndcg": all_results[5],
            "f1": 0,  # Not used
            "total": all_results[6],
        }
        
        perf = get_performance_dict(return_dict)
        
        return perf, all_targets, all_predictions
    
    def print_results(self, perf):
        """Print evaluation results."""
        print("\n" + "="*50)
        print("TEST SET RESULTS")
        print("="*50)
        print(f"Acc@1:  {perf['acc@1']:.2f}%")
        print(f"Acc@5:  {perf['acc@5']:.2f}%")
        print(f"Acc@10: {perf['acc@10']:.2f}%")
        print(f"MRR:    {perf['mrr']:.2f}%")
        print(f"NDCG:   {perf['ndcg']:.2f}%")
        print(f"Total:  {int(perf['total'])}")
        print("="*50)
