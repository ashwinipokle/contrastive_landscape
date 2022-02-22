import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import SymReLU

class SparseContrastiveModelReduced(nn.Module):
    def __init__(self, Wo_init, Wt_init, m, p, d,
                    has_online_ReLU=True,
                    has_target_ReLU=True,
                    device=None,
                    normalize_rep=True,
                    use_bn=False,
                    use_pred=False,
                    linear_pred=False,
                    use_bias=True,
                    const_bias=False,
                    const_bias_val=1,
                    ) -> None:

        super().__init__()
        self.p=p
        self.m=m
        self.d=d

        self.Wo = nn.Linear(p, m, bias=use_bias)
        self.Wt = nn.Linear(p, m, bias=use_bias)

        if not use_bias and const_bias:
            print("Bias will be a constant!!!")
            self.bo = nn.Parameter(torch.ones(m) * const_bias_val, requires_grad=False)
            self.bt = nn.Parameter(torch.ones(m) * const_bias_val, requires_grad=False)
        
        self.const_bias = const_bias

        self.srelu = SymReLU()

        self.init_weights(Wo_init, Wt_init)

        self.device = device

        self.normalize_rep = normalize_rep
        self.name = "simplified"

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

        if self.const_bias:
            zo += self.bo
            zt += self.bt 

        self.zo = zo
        self.zt = zt

        if self.const_bias:
            zo = self.srelu(zo, self.bo)
            zt = self.srelu(zt, self.bt)
        else:
            zo = self.srelu(zo, self.Wo.bias)
            zt = self.srelu(zt, self.Wt.bias)

        self.error = zo - zt

        self.predicted_rep = zo
        self.target_rep = zt

        if self.normalize_rep:
            zt = F.normalize(zt, dim=1)
            zo = F.normalize(zo, dim=1)

        if optimize_online:
            #loss = torch.sum((zo - zt.detach())**2, dim=-1).mean()
            loss = 2 - 2 * (zo * zt.detach()).sum(dim=1).mean()
        else:
            #loss = torch.sum((zo.detach() - zt)**2, dim=-1).mean()
            loss = 2 - 2 * (zo.detach() * zt).sum(dim=1).mean()

        if not self.const_bias:
            self.bo = self.Wo.bias
            self.bt = self.Wt.bias

        return loss
