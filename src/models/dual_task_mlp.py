"""
Dual-Task MLP Model

Based on research paper: "Dual-Task MLP on Behavioral Alpha Factors"

Architecture:
- Input: 40 behavioral factors
- Hidden: 64 -> 32 with BatchNorm + Dropout
- Output: Regression head (5-day return) + Classification head (up/down)
- Loss: MSE + 0.5 * BCE
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import os


class DualTaskMLP(nn.Module):
    """
    Dual-output MLP for stock prediction.
    
    Predicts both:
    1. Regression: 5-day forward return (continuous)
    2. Classification: Up/down direction (binary)
    """
    
    def __init__(self, input_dim: int = 40, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.1):
        super(DualTaskMLP, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(dropout)
        
        # Regression head: predict 5-day forward return
        self.reg_head = nn.Linear(hidden2, 1)
        
        # Classification head: predict up/down
        self.clf_head = nn.Linear(hidden2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 40)
            
        Returns:
            (reg_out, clf_out) where reg_out is continuous and clf_out is probability
        """
        # Shared feature extraction
        h1 = torch.relu(self.bn1(self.fc1(x)))
        h1 = self.dropout1(h1)
        
        h2 = torch.relu(self.bn2(self.fc2(h1)))
        h2 = self.dropout2(h2)
        
        # Outputs
        reg_out = self.reg_head(h2)  # No activation; continuous return
        clf_out = self.sigmoid(self.clf_head(h2))  # Sigmoid for probability
        
        return reg_out, clf_out
    
    def get_deep_alpha(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute combined score for ranking.
        deep_alpha = regression_output × classification_probability
        
        Args:
            x: Input tensor (batch, 40)
            
        Returns:
            Combined score for ranking stocks
        """
        reg_out, clf_out = self.forward(x)
        deep_alpha = reg_out.squeeze() * clf_out.squeeze()
        return deep_alpha


class DualTaskTrainer:
    """
    Training pipeline for DualTaskMLP.
    """
    
    def __init__(self, 
                 model: DualTaskMLP,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 1e-3,
                 clf_loss_weight: float = 0.5,
                 device: str = 'cpu'):
        
        self.model = model.to(device)
        self.device = device
        self.clf_loss_weight = clf_loss_weight
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.7, 
            patience=5
        )
        
        self.criterion_reg = nn.MSELoss()
        self.criterion_clf = nn.BCELoss()
        
    def train_epoch(self, X: np.ndarray, y_reg: np.ndarray, y_clf: np.ndarray, 
                    batch_size: int = 2048) -> float:
        """Train for one epoch."""
        self.model.train()
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y_reg = y_reg[indices]
        y_clf = y_clf[indices]
        
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(X), batch_size):
            batch_X = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(self.device)
            batch_y_reg = torch.tensor(y_reg[i:i+batch_size], dtype=torch.float32).to(self.device)
            batch_y_clf = torch.tensor(y_clf[i:i+batch_size], dtype=torch.float32).to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            reg_out, clf_out = self.model(batch_X)
            
            # Losses
            loss_reg = self.criterion_reg(reg_out.squeeze(), batch_y_reg)
            loss_clf = self.criterion_clf(clf_out.squeeze(), batch_y_clf)
            loss = loss_reg + self.clf_loss_weight * loss_clf
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def validate(self, X: np.ndarray, y_reg: np.ndarray, y_clf: np.ndarray) -> Tuple[float, float]:
        """Validate on data."""
        self.model.eval()
        
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_reg_t = torch.tensor(y_reg, dtype=torch.float32).to(self.device)
            y_clf_t = torch.tensor(y_clf, dtype=torch.float32).to(self.device)
            
            reg_out, clf_out = self.model(X_t)
            
            loss_reg = self.criterion_reg(reg_out.squeeze(), y_reg_t)
            loss_clf = self.criterion_clf(clf_out.squeeze(), y_clf_t)
            
            # Classification accuracy
            preds = (clf_out.squeeze() > 0.5).float()
            acc = (preds == y_clf_t).float().mean().item()
        
        return loss_reg.item() + self.clf_loss_weight * loss_clf.item(), acc
    
    def fit(self, 
            X_train: np.ndarray, y_reg_train: np.ndarray, y_clf_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, 
            y_reg_val: Optional[np.ndarray] = None,
            y_clf_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            patience: int = 15,
            batch_size: int = 2048,
            verbose: bool = True) -> dict:
        """
        Full training loop with early stopping.
        
        Returns:
            Training history
        """
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(X_train, y_reg_train, y_clf_train, batch_size)
            history['train_loss'].append(train_loss)
            
            # Validate
            if X_val is not None:
                val_loss, val_acc = self.validate(X_val, y_reg_val, y_clf_val)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                self.scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                          f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}")
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return history
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions.
        
        Returns:
            (reg_predictions, clf_probabilities, deep_alpha_scores)
        """
        self.model.eval()
        
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            reg_out, clf_out = self.model(X_t)
            deep_alpha = reg_out.squeeze() * clf_out.squeeze()
        
        return (reg_out.squeeze().cpu().numpy(), 
                clf_out.squeeze().cpu().numpy(),
                deep_alpha.cpu().numpy())
    
    def save(self, path: str):
        """Save model."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")


def compute_ic(predictions: np.ndarray, realized_returns: np.ndarray) -> float:
    """
    Compute Spearman rank correlation (Information Coefficient).
    """
    from scipy.stats import spearmanr
    
    # Handle NaN
    valid_mask = ~(np.isnan(predictions) | np.isnan(realized_returns))
    if valid_mask.sum() < 10:
        return 0.0
    
    ic, _ = spearmanr(predictions[valid_mask], realized_returns[valid_mask])
    return ic if not np.isnan(ic) else 0.0


if __name__ == "__main__":
    # Quick test
    model = DualTaskMLP(input_dim=40)
    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    x = torch.randn(32, 40)
    reg, clf = model(x)
    print(f"Regression output: {reg.shape}")
    print(f"Classification output: {clf.shape}")
    print(f"Deep alpha: {model.get_deep_alpha(x).shape}")
