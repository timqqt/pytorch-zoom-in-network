import torch

def MultinomialRegularizer(x, strength, eps=1e-6):
    logx = torch.log(x + eps)
    return strength * torch.sum(x * logx) / float(x.shape[0])
