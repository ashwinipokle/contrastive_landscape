import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *

class SimSiamMultiLayeredModel(nn.Module):
    def __init__(self, Wo_init, Wo1_init, m, p, d,
                    has_online_ReLU=True,
                    has_target_ReLU=True,
                    device=None,
                    use_bn=None,
                    ) -> None:

        super().__init__()
        self.p=p
        self.m=m
        self.d=d

        self.Wo = nn.Linear(p, m, bias=True)
        self.Wo_1 = nn.Linear(m, m, bias=True)
        self.Wp = nn.Linear(m, m, bias=True)

        self.srelu = SymReLU()

        self.has_online_ReLU = has_online_ReLU
        self.has_target_ReLU = has_target_ReLU

        self.init_weights(Wo_init, Wo1_init)

        self.device = device
        self.use_bn = use_bn
        if use_bn is not None:
            self.norm1 = nn.BatchNorm1d(m)
            self.norm2 = nn.BatchNorm1d(m)
            print(f"Using batch norm {use_bn}")
        self.name = "simsiam-ml"

    def init_weights(self, Wo_init, Wo1_init):
        if self.Wo.weight.shape == Wo_init.T.shape:
            Wo_init = Wo_init.T
            Wo1_init = Wo1_init.T

        assert Wo_init.shape == self.Wo.weight.shape
        assert Wo1_init.shape == self.Wo_1.weight.shape, f"{Wo1_init.shape} {self.Wo_1.weight.shape}"

        with torch.no_grad():
            self.Wo.weight.data = torch.from_numpy(Wo_init).type(torch.float)
            self.Wo_1.weight.data = torch.from_numpy(Wo1_init).type(torch.float)

    def forward(self, x1, x2):
        zo, zt = self.Wo(x1), self.Wo(x2)
        if self.use_bn:
            zo = self.norm1(zo)
            zt = self.norm1(zt)

        zo = self.srelu(zo, self.Wo.bias)
        zt = self.srelu(zt, self.Wo.bias)

        zo, zt = self.Wo_1(zo), self.Wo_1(zt)
        if self.use_bn:
            zo = self.norm2(zo)
            zt = self.norm2(zt)

        zo = self.srelu(zo, self.Wo_1.bias)
        zt = self.srelu(zt, self.Wo_1.bias)

        self.predicted_rep = zo  # Used for checking support in pytorch_utils.py
        self.target_rep = zt

        p1, p2 = self.Wp(zo), self.Wp(zt)

        loss = 0.5 * (D(p1, zt, "simplified") + D(p2, zo, "simplified"))
        return loss