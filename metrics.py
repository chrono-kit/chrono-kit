import torch

class Metric:
    def __init__(self):
        pass
    
    @staticmethod
    def mae(y_pred, y_true):
        
        return torch.mean(torch.abs(torch.sub(y_pred, y_true)))

    @staticmethod
    def mse(y_pred, y_true):
        
        return torch.mean((torch.square(torch.sub(y_pred, y_true))))

    @staticmethod
    def rmse(y_pred, y_true):
        
        return torch.sqrt(torch.mean((torch.square(torch.sub(y_pred, y_true)))))
