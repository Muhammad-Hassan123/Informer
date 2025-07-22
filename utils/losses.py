import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GMADLLoss(nn.Module):
    """
    Generalized Mean Absolute Deviation Loss (GMADL)
    
    This loss function is particularly useful for time series forecasting as it:
    1. Is more robust to outliers than MSE
    2. Provides better gradient flow than pure MAE
    3. Can be tuned with the beta parameter for different sensitivities
    
    Args:
        beta (float): Shape parameter controlling the loss behavior
                     - beta = 1.0: Standard MAE
                     - beta = 2.0: Standard MSE  
                     - beta between 1-2: Balanced between MAE and MSE
        reduction (str): Specifies the reduction to apply to the output
    """
    def __init__(self, beta=1.5, reduction='mean'):
        super(GMADLLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        
    def forward(self, pred, true):
        """
        Calculate GMADL loss
        
        Args:
            pred: Predicted values [batch_size, seq_len, features]
            true: True values [batch_size, seq_len, features]
            
        Returns:
            loss: GMADL loss value
        """
        # Calculate absolute differences
        abs_diff = torch.abs(pred - true)
        
        # Apply generalized mean with beta parameter
        # GMADL = (|pred - true|^beta)^(1/beta)
        # For numerical stability, we use: beta * log(|pred - true| + eps)
        eps = 1e-8
        loss = torch.pow(abs_diff + eps, self.beta)
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class AdaptiveGMADLLoss(nn.Module):
    """
    Adaptive GMADL that adjusts beta based on training progress
    This can help with convergence by starting more robust (lower beta) 
    and becoming more precise (higher beta) as training progresses.
    """
    def __init__(self, beta_start=1.2, beta_end=1.8, total_epochs=20, reduction='mean'):
        super(AdaptiveGMADLLoss, self).__init__()
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_epochs = total_epochs
        self.reduction = reduction
        self.current_epoch = 0
        
    def set_epoch(self, epoch):
        """Update the current epoch for adaptive beta calculation"""
        self.current_epoch = epoch
        
    def get_current_beta(self):
        """Calculate current beta based on training progress"""
        if self.total_epochs <= 1:
            return self.beta_end
        
        progress = min(self.current_epoch / (self.total_epochs - 1), 1.0)
        beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        return beta
        
    def forward(self, pred, true):
        current_beta = self.get_current_beta()
        abs_diff = torch.abs(pred - true)
        eps = 1e-8
        loss = torch.pow(abs_diff + eps, current_beta)
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class WeightedGMADLLoss(nn.Module):
    """
    Weighted GMADL that gives different importance to different time steps
    Useful for crypto forecasting where recent predictions might be more important
    """
    def __init__(self, beta=1.5, weight_decay=0.95, reduction='mean'):
        super(WeightedGMADLLoss, self).__init__()
        self.beta = beta
        self.weight_decay = weight_decay
        self.reduction = reduction
        
    def forward(self, pred, true):
        """
        Args:
            pred: [batch_size, seq_len, features]
            true: [batch_size, seq_len, features]
        """
        abs_diff = torch.abs(pred - true)
        eps = 1e-8
        
        # Create exponentially decaying weights (more weight on later time steps)
        seq_len = pred.shape[1]
        weights = torch.tensor([self.weight_decay ** (seq_len - i - 1) for i in range(seq_len)])
        weights = weights.to(pred.device).view(1, -1, 1)  # [1, seq_len, 1]
        
        # Apply GMADL with weights
        loss = weights * torch.pow(abs_diff + eps, self.beta)
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

def get_loss_function(loss_name, **kwargs):
    """
    Factory function to get loss functions
    
    Args:
        loss_name (str): Name of the loss function
        **kwargs: Additional arguments for the loss function
        
    Returns:
        loss_fn: The loss function
    """
    if loss_name.lower() == 'mse':
        return nn.MSELoss()
    elif loss_name.lower() == 'mae':
        return nn.L1Loss()
    elif loss_name.lower() == 'gmadl':
        beta = kwargs.get('beta', 1.5)
        return GMADLLoss(beta=beta)
    elif loss_name.lower() == 'adaptive_gmadl':
        beta_start = kwargs.get('beta_start', 1.2)
        beta_end = kwargs.get('beta_end', 1.8)
        total_epochs = kwargs.get('total_epochs', 20)
        return AdaptiveGMADLLoss(beta_start, beta_end, total_epochs)
    elif loss_name.lower() == 'weighted_gmadl':
        beta = kwargs.get('beta', 1.5)
        weight_decay = kwargs.get('weight_decay', 0.95)
        return WeightedGMADLLoss(beta=beta, weight_decay=weight_decay)
    elif loss_name.lower() == 'huber':
        delta = kwargs.get('delta', 1.0)
        return nn.HuberLoss(delta=delta)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")