import torch
from chronokit.preprocessing.dataloader import DataLoader

"""Performance evaluation metrics for model predictions"""

def mae(y_pred, y_true):
    """
    Mean Absolute Error
    
    Arguments:
    
    *y_pred (array_like): Predicted values
    *y_true (array_like): Ground truth values
    """
    y_pred = DataLoader(y_pred).to_tensor()
    y_true = DataLoader(y_true).to_tensor()      

    return torch.mean(torch.abs(torch.sub(y_pred, y_true)))


def mse(y_pred, y_true):
    """
    Mean Squared Error
    
    Arguments:
    
    *y_pred (array_like): Predicted values
    *y_true (array_like): Ground truth values
    """
    y_pred = DataLoader(y_pred).to_tensor()
    y_true = DataLoader(y_true).to_tensor()      

    return torch.mean((torch.square(torch.sub(y_pred, y_true))))

def rmse(y_pred, y_true):
    """
    Root Mean Squared Error
    
    Arguments:
    
    *y_pred (array_like): Predicted values
    *y_true (array_like): Ground truth values
    """
    y_pred = DataLoader(y_pred).to_tensor()
    y_true = DataLoader(y_true).to_tensor()  

    return torch.sqrt(torch.mean((torch.square(torch.sub(y_pred, y_true)))))