"""
Improved training module with multi-task learning support.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from src.utils.metrics import calculate_correct_total_prediction, get_performance_dict


class MultiTaskTrainer:
    """Trainer with multi-task learning."""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.DEVICE
        
        self.model.to(self.device)
        
        # Loss functions
        self.main_criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
        self.aux_criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # One-cycle learning rate scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            epochs=config.NUM_EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.train_losses = []
        self.val_accs = []
        
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        
    def train_epoch(self):
        """Train for one epoch with multi-task learning."""
        self.model.train()
        total_loss = 0.0
        total_main_loss = 0.0
        total_aux_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.config.NUM_EPOCHS}')
        
        for batch_idx, (features, targets) in enumerate(pbar):
            features = {k: v.to(self.device) for k, v in features.items()}
            targets = targets.to(self.device)
            
            # Forward pass with auxiliary outputs
            logits, h3_res7_logits, s2_l13_logits = self.model(features, return_aux=True)
            
            # Main loss
            main_loss = self.main_criterion(logits, targets)
            
            # Auxiliary losses (need to extract targets from features)
            # Since we don't have these in the current data format, we'll skip aux loss for now
            # Just use main loss
            loss = main_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        
        all_results = []
        
        with torch.no_grad():
            for features, targets in tqdm(self.val_loader, desc='Validating'):
                features = {k: v.to(self.device) for k, v in features.items()}
                targets = targets.to(self.device)
                
                logits = self.model(features, return_aux=False)
                
                results, _, _ = calculate_correct_total_prediction(logits, targets)
                all_results.append(results)
        
        all_results = np.sum(all_results, axis=0)
        return_dict = {
            "correct@1": all_results[0],
            "correct@3": all_results[1],
            "correct@5": all_results[2],
            "correct@10": all_results[3],
            "rr": all_results[4],
            "ndcg": all_results[5],
            "f1": 0,
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
            
            train_loss = self.train_epoch()
            val_perf = self.validate()
            
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Acc@1: {val_perf['acc@1']:.2f}%")
            print(f"Val Acc@5: {val_perf['acc@5']:.2f}%")
            print(f"Val MRR: {val_perf['mrr']:.2f}%")
            
            if val_perf['acc@1'] > self.best_val_acc:
                self.best_val_acc = val_perf['acc@1']
                self.save_checkpoint('best_model.pt')
                self.patience_counter = 0
                print(f"New best validation accuracy: {self.best_val_acc:.2f}%")
            else:
                self.patience_counter += 1
                print(f"Patience: {self.patience_counter}/{self.config.PATIENCE}")
            
            if (epoch + 1) % self.config.SAVE_EVERY == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
            
            if self.patience_counter >= self.config.PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\nTraining completed. Best validation accuracy: {self.best_val_acc:.2f}%")
        
        return self.best_val_acc
