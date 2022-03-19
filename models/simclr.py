import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import SymReLU
from models.nt_xent import NT_Xent

class SimCLROrigModel(nn.Module):
    def __init__(self, Wo_init, m, p, d, 
                    has_online_ReLU=True,
                    has_target_ReLU=True,
                    device=None,
                    batch_size=64,
                    temperature=0.05
                    ) -> None:
        
        super().__init__()
        self.p=p
        self.m=m
        self.d=d

        self.Wo = nn.Linear(p, m, bias=True)
        self.Wp = nn.Linear(m, m, bias=True)

        self.srelu = SymReLU()

        self.has_online_ReLU = has_online_ReLU
        self.has_target_ReLU = has_target_ReLU

        self.init_weights(Wo_init)

        self.criterion = NT_Xent(batch_size, temperature)

        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.bn1 = nn.BatchNorm1d(m)

        self.name = "simclr-orig"
    
    def init_weights(self, Wo_init):
        if self.Wo.weight.shape == Wo_init.T.shape:
            Wo_init = Wo_init.T
        
        assert Wo_init.shape == self.Wo.weight.shape

        with torch.no_grad():
            self.Wo.weight.data = torch.from_numpy(Wo_init).type(torch.float)

    def forward(self, x1, x2):
        zo, zt = self.Wo(x1), self.Wo(x2)

        zo = self.srelu(self.bn1(zo), self.Wo.bias)
        zt = self.srelu(self.bn1(zt), self.Wo.bias)
        
        self.predicted_rep = zo
        self.target_rep = zt

        zo = self.Wp(zo)
        zt = self.Wp(zt)

        loss = self.criterion(zo, zt)
        return loss

class SimCLRModel(nn.Module):
    def __init__(self, Wo_init, m, p, d, 
                    has_online_ReLU=True,
                    has_target_ReLU=True,
                    device=None,
                    batch_size=64,
                    temperature=0.05,
                    use_bn=False,
                    ) -> None:
        
        super().__init__()
        self.p=p
        self.m=m
        self.d=d

        self.Wo = nn.Linear(p, m, bias=True)

        self.srelu = SymReLU()

        self.has_online_ReLU = has_online_ReLU
        self.has_target_ReLU = has_target_ReLU

        self.use_bn = use_bn
        self.bn1 = nn.BatchNorm1d(m)

        self.init_weights(Wo_init)

        self.criterion = NT_Xent(batch_size, temperature)

        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.name = "simclr"
    
    def init_weights(self, Wo_init):
        if self.Wo.weight.shape == Wo_init.T.shape:
            Wo_init = Wo_init.T
        
        assert Wo_init.shape == self.Wo.weight.shape

        with torch.no_grad():
            self.Wo.weight.data = torch.from_numpy(Wo_init).type(torch.float)

    def forward(self, x1, x2):
        zo, zt = self.Wo(x1), self.Wo(x2)

        if self.use_bn:
            zo = self.bn1(zo)
            zt = self.bn1(zt)
        
        if self.has_online_ReLU and self.has_target_ReLU:
            zo = self.srelu(zo, self.Wo.bias)
            zt = self.srelu(zt, self.Wo.bias)
        
        self.predicted_rep = zo
        self.target_rep = zt

        loss = self.criterion(zo, zt)
        return loss