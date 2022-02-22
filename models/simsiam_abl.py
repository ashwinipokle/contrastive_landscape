import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import *

class SimSiamAblationModel(nn.Module):
    def __init__(self, Wo_init, m, p, d,
                    has_online_ReLU=True,
                    has_target_ReLU=True,
                    device=None,
                    batch_norm=None,
                    use_bn=True,
                    use_diag_pred=False
                    ) -> None:

        super().__init__()
        self.p=p
        self.m=m
        self.d=d

        self.Wo = nn.Linear(p, m, bias=True)
        if use_diag_pred:
            k = 1/np.sqrt(m)
            self.Wp = nn.Parameter((2*torch.rand((m, 1), device=device) - 1)*k, requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(m, device=device), requires_grad=True)
        else:
            self.Wp = nn.Linear(m, m, bias=True)

        self.use_diag_pred = use_diag_pred
        self.srelu = SymReLU()

        self.has_online_ReLU = has_online_ReLU
        self.has_target_ReLU = has_target_ReLU

        self.init_weights(Wo_init)

        self.device = device
        self.use_bn = use_bn
        self.batch_norm = batch_norm

        if batch_norm is not None or self.use_bn:
            self.norm1 = nn.BatchNorm1d(m)

            print(f"Using batch norm {batch_norm}")
        self.name = "simsiam-diag"

    def init_weights(self, Wo_init):
        if self.Wo.weight.shape == Wo_init.T.shape:
            Wo_init = Wo_init.T

        assert Wo_init.shape == self.Wo.weight.shape

        with torch.no_grad():
            self.Wo.weight.data = torch.from_numpy(Wo_init).type(torch.float)

    def forward(self, x1, x2):
        zo, zt = self.Wo(x1), self.Wo(x2)
        if self.use_bn:
            zo = self.norm1(zo)
            zt = self.norm1(zt)

        zo = self.srelu(zo, self.Wo.bias)
        zt = self.srelu(zt, self.Wo.bias)

        self.predicted_rep = zo  # Used for checking support in pytorch_utils.py
        self.target_rep = zt

        if self.use_diag_pred:
            p1, p2 = zo @ self.Wp + self.bias, zt @ self.Wp + self.bias
        else:
            p1, p2 = self.Wp(zo), self.Wp(zt)

        #loss = 0.5 * (D(zo, zt, "simplified") + D(zt, zo, "simplified"))
        loss = 0.5 * (D(p1, zt, "simplified") + D(p2, zo, "simplified"))
        return loss