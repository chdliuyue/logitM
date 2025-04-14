import torch


def compute_accuracy(predictions, targets):
    _, preds = torch.max(predictions, dim=1)
    correct = (preds == targets).sum().item()
    return correct



