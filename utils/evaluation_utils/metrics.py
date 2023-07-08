import torch
from preprocessing.dataloader import DataLoader

def mae(y_pred, y_true):
    
    y_pred = DataLoader(y_pred).to_tensor()
    y_true = DataLoader(y_true).to_tensor()      

    return torch.mean(torch.abs(torch.sub(y_pred, y_true)))


def mse(y_pred, y_true):

    y_pred = DataLoader(y_pred).to_tensor()
    y_true = DataLoader(y_true).to_tensor()      

    return torch.mean((torch.square(torch.sub(y_pred, y_true))))

def rmse(y_pred, y_true):

    y_pred = DataLoader(y_pred).to_tensor()
    y_true = DataLoader(y_true).to_tensor()  

    return torch.sqrt(torch.mean((torch.square(torch.sub(y_pred, y_true)))))