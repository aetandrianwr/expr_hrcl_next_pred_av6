"""
Training module for next-location prediction.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src.utils.metrics import calculate_correct_total_prediction, get_performance_dict


class Trainer:
    """Trainer class for the model."""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.DEVICE
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler with warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.WARMUP_STEPS
        )
        
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS * len(train_loader) - config.WARMUP_STEPS,
            eta_min=config.LEARNING_RATE * 0.01
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[config.WARMUP_STEPS]
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.train_losses = []
        self.val_accs = []
        
        # Create checkpoint directory
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.config.NUM_EPOCHS}')
        
        for batch_idx, (features, targets) in enumerate(pbar):
            # Move to device
            features = {k: v.to(self.device) for k, v in features.items()}
            targets = targets.to(self.device)
            
            # Forward pass
            logits = self.model(features)
            loss = self.criterion(logits, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'lr': self.optimizer.param_groups[0]['lr']})
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        
        all_results = []
        
        with torch.no_grad():
            for features, targets in tqdm(self.val_loader, desc='Validating'):
                # Move to device
                features = {k: v.to(self.device) for k, v in features.items()}
                targets = targets.to(self.device)
                
                # Forward pass
                logits = self.model(features)
                
                # Calculate metrics
                results, _, _ = calculate_correct_total_prediction(logits, targets)
                all_results.append(results)
        
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
        self.val_accs.append(perf['acc@1'])
        
        return perf
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_accs': self.val_accs,
        }
        
        path = os.path.join(self.config.CHECKPOINT_DIR, filename)
        torch.save(checkpoint, path)
        print(f'Checkpoint saved to {path}')
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        path = os.path.join(self.config.CHECKPOINT_DIR, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint['train_losses']
        self.val_accs = checkpoint['val_accs']
        
        print(f'Checkpoint loaded from {path}')
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on device: {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        for epoch in range(self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_perf = self.validate()
            
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Acc@1: {val_perf['acc@1']:.2f}%")
            print(f"Val Acc@5: {val_perf['acc@5']:.2f}%")
            print(f"Val MRR: {val_perf['mrr']:.2f}%")
            
            # Save best model
            if val_perf['acc@1'] > self.best_val_acc:
                self.best_val_acc = val_perf['acc@1']
                self.save_checkpoint('best_model.pt')
                self.patience_counter = 0
                print(f"New best validation accuracy: {self.best_val_acc:.2f}%")
            else:
                self.patience_counter += 1
                print(f"Patience: {self.patience_counter}/{self.config.PATIENCE}")
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.SAVE_EVERY == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
            
            # Early stopping
            if self.patience_counter >= self.config.PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\nTraining completed. Best validation accuracy: {self.best_val_acc:.2f}%")
        
        return self.best_val_acc
