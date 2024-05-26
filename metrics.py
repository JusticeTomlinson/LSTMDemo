import torch

def calc_rmse(predictions, targets):
    return torch.sqrt(((predictions - targets) ** 2).mean())

def calc_mae(predictions, targets):
    return torch.abs(predictions - targets).mean()

def calc_mape(predictions, targets):
    print(predictions)
    print(targets)
    return torch.mean(torch.abs((targets - predictions) / targets)) * 100