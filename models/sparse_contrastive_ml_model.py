import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import SymReLU

class SparseContrastiveMultiLayeredModel(nn.Module):
    def __init__(self, Wo_init, Wt_init, Wo1_init, Wt1_init, m, p, d,
                    has_online_ReLU=True,
                    has_target_ReLU=True,
                    device=None,
                    normalize_rep=True,
                    use_bn=False,
                    ) -> None:

        super().__init__()
        self.p=p
        self.m=m
        self.d=d

        self.use_bn = use_bn
        self.Wo = nn.Linear(p, m, bias=True)
        self.Wo_1 = nn.Linear(m, m, bias=True)

        self.Wt = nn.Linear(p, m, bias=True)
        self.Wt_1 = nn.Linear(m, m, bias=True)

        if use_bn:
            self.bno = nn.BatchNorm1d(m)
            self.bnt = nn.BatchNorm1d(m)

            self.bno_1 = nn.BatchNorm1d(m)
            self.bnt_1 = nn.BatchNorm1d(m)

        print(self.Wo.weight.shape)
        if has_online_ReLU and has_target_ReLU:
            self.srelu = SymReLU()

        self.has_online_ReLU = has_online_ReLU
        self.has_target_ReLU = has_target_ReLU

        self.init_weights(Wo_init, Wt_init, Wo1_init, Wt1_init)

        self.device = device

        self.normalize_rep = normalize_rep
        self.name = "simplified-ml"

    def init_weights(self, Wo_init, Wt_init, Wo1_init, Wt1_init):
        if self.Wo.weight.shape == Wo_init.T.shape:
            Wo_init = Wo_init.T
            Wt_init = Wt_init.T

            Wo1_init = Wo1_init.T
            Wt1_init = Wt1_init.T

        assert Wo_init.shape == self.Wo.weight.shape, f"{Wo_init.shape} {self.Wo.weight.shape}"
        assert Wt_init.shape == self.Wt.weight.shape, f"{Wt_init.shape} {self.Wt.weight.shape}"
        assert Wo1_init.shape == self.Wo_1.weight.shape, f"{Wo1_init.shape} {self.Wo_1.weight.shape}"
        assert Wt1_init.shape == self.Wt_1.weight.shape, f"{Wt1_init.shape} {self.Wt_1.weight.shape}"

        with torch.no_grad():
            self.Wo.weight.data = torch.from_numpy(Wo_init).type(torch.float)
            self.Wt.weight.data = torch.from_numpy(Wt_init).type(torch.float)
            self.Wo_1.weight.data = torch.from_numpy(Wo1_init).type(torch.float)
            self.Wt_1.weight.data = torch.from_numpy(Wt1_init).type(torch.float)

    def forward(self, x1, x2, optimize_online=True):
        zo = self.Wo(x1)
        zt = self.Wt(x2)

        if self.use_bn:
            zo = self.bno(zo)
            zt = self.bnt(zt)

        zo = self.srelu(zo, self.Wo.bias)
        zt = self.srelu(zt, self.Wt.bias)

        zo = self.Wo_1(zo)
        zt = self.Wt_1(zt)

        self.zo = zo
        self.zt = zt

        if self.use_bn:
            zo = self.bno_1(zo)
            zt = self.bnt_1(zt)

        zo = self.srelu(zo, self.Wo_1.bias)
        zt = self.srelu(zt, self.Wt_1.bias)

        self.error = zo - zt

        self.predicted_rep = zo
        self.target_rep = zt

        if self.normalize_rep:
            zo = F.normalize(zo, dim=1)
            zt = F.normalize(zt, dim=1)

        #loss = torch.sum((zo - zt)**2, dim=-1).mean()
        if optimize_online:
            loss = torch.sum((zo - zt.detach())**2, dim=-1).mean()
        else:
            loss = torch.sum((zo.detach() - zt)**2, dim=-1).mean()

        self.bo = self.Wo.bias
        self.bt = self.Wt.bias
        return loss