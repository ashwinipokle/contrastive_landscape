import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import SymReLU

class SparseContrastiveModel(nn.Module):
    def __init__(self, Wo_init, Wt_init, m, p, d,
                    has_online_ReLU=True,
                    has_target_ReLU=True,
                    device=None,
                    normalize_rep=True,
                    use_bn=False,
                    use_pred=False,
                    linear_pred=False,
                    use_pred_bias=True
                    ) -> None:

        super().__init__()
        self.p=p
        self.m=m
        self.d=d

        self.use_bn = use_bn
        self.use_pred = use_pred
        self.linear_pred = linear_pred

        self.Wo = nn.Linear(p, m, bias=True)
        self.Wt = nn.Linear(p, m, bias=True)
        # No bias in predictor
        if use_pred:
            self.Wp = nn.Linear(m, m, bias=use_pred_bias)

        if use_bn:
            self.bno = nn.BatchNorm1d(m)
            self.bnt = nn.BatchNorm1d(m)

        if has_online_ReLU and has_target_ReLU:
            self.srelu = SymReLU()

        self.has_online_ReLU = has_online_ReLU
        self.has_target_ReLU = has_target_ReLU

        self.init_weights(Wo_init, Wt_init)

        self.device = device

        self.normalize_rep = normalize_rep
        self.name = "simplified-pred"

    def init_weights(self, Wo_init, Wt_init):
        if self.Wo.weight.shape == Wo_init.T.shape:
            Wo_init = Wo_init.T
            Wt_init = Wt_init.T

        assert Wo_init.shape == self.Wo.weight.shape
        assert Wt_init.shape == self.Wt.weight.shape

        with torch.no_grad():
            self.Wo.weight.data = torch.from_numpy(Wo_init).type(torch.float)
            self.Wt.weight.data = torch.from_numpy(Wt_init).type(torch.float)

    def forward(self, x1, x2, optimize_online=True):
        zo = self.Wo(x1)
        zt = self.Wt(x2)

        self.zo = zo
        self.zt = zt

        if self.use_bn:
            zo = self.bno(zo)
            zt = self.bnt(zt)

        if self.has_online_ReLU and self.has_target_ReLU:
            zo = self.srelu(zo, self.Wo.bias)
            zt = self.srelu(zt, self.Wt.bias)

        self.error = zo - zt

        self.predicted_rep = zo
        self.target_rep = zt

        if self.use_pred:
            if optimize_online:
                po = self.Wp(zo)
                if self.normalize_rep:
                    po = F.normalize(po, dim=1)
            else:
                pt = self.Wp(zt)
                if self.normalize_rep:
                    pt = F.normalize(pt, dim=1)
        
        if self.normalize_rep:
            if optimize_online:
                zt = F.normalize(zt, dim=1)
            else:
                zo = F.normalize(zo, dim=1)
            
        if optimize_online:
            if self.use_pred:
                loss = torch.sum((po - zt.detach())**2, dim=-1).mean()
                #loss = 2 - 2 * (po * zt.detach()).sum(dim=1).mean()
            else:
                loss = torch.sum((zo - zt.detach())**2, dim=-1).mean()

        else:
            if self.use_pred:
                loss = torch.sum((zo.detach() - pt)**2, dim=-1).mean()
                #loss = 2 - 2 * (zo.detach() * pt).sum(dim=1).mean()
            else:
                loss = torch.sum((zo.detach() - zt)**2, dim=-1).mean()

        self.bo = self.Wo.bias
        self.bt = self.Wt.bias
        return loss