import torch

def charbonier_loss(x, epsilon):
    loss = torch.mean(torch.sqrt(x * x + epsilon * epsilon))
    return loss