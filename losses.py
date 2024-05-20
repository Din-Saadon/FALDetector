import torch

from torch.nn.modules.module import Module
from packages.resample2d_package.resample2d import Resample2d
from packages.grad_package.grad import Grad

class L1(Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

class L2(Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

class L_epe(Module):
    def __init__(self):
        super(L_epe, self).__init__()
        self.norm = L2()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue
        
class L_ms(Module):
    def __init__(self):
        super(L_ms, self).__init__()
        self.norm = L2()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue
        
class L_rec(Module):
    def __init__(self,flow_factor):
        super(L_epe, self).__init__()
        self.norm = L1()
        self.flow_factor = -1*flow_factor
    def forward(self, fx, Xo,Xm):
        resample = Resample2d()
        fx = fx*self.flow_factor
        Xo_like = resample(Xm,fx)
        lossvalue = self.norm(Xo_like,Xo)
        return lossvalue

class L_tot(Module):
    def __init__(self,s_epe,s_ms,s_rec,flow_factor):
        super(L_epe,self).__init__()
        self.s_epe = s_epe
        self.s_ms = s_ms
        self.s_rec = s_rec
        self.loss_epe = L_epe()
        self.loss_ms = L_ms
        self.loss_rec = L_rec(flow_factor)
    def forward(self, fx, Xo , Xm ,Umo , M):
        return self.s_epe*self.loss_epe(fx,M,Umo)+self.s_ms*self.loss_ms(fx,M,Umo)+self.s_rec*self.loss_rec(fx,Xo,Xm)