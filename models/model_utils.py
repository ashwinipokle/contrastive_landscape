import torch
import torch.nn as nn
import torch.nn.functional as F

class SymReLU(nn.Module):
    def __init__(self, use_neg_bias=False) -> None:
        super().__init__()
        self.use_neg_bias = use_neg_bias

    def forward(self, x, b):
        if self.use_neg_bias:
            return F.relu(x) - F.relu(-x - 2*b)
        return F.relu(x) - F.relu(-x + 2*b)

def D(p, z, version='l2'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

    elif version == 'l2':
        z = z.detach()
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return torch.sum((p - z)**2, dim=-1).mean()
    else:
        raise Exception
