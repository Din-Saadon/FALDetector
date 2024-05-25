import torch

from torch.nn.modules.module import Module
from packages.resample2d_package.resample2d import Resample2d
from packages.grad_package.grad import Grad

def _m_resize(M,s):
    assert(M.dim()==4)
    _ , _ , h , w = M.shape
    return   M[:,:,:,s:w-s] , M[:,:,s:h-s,:]

class L1(Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target=0):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

class L2(Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target=0):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

class L_epe(Module):
    def __init__(self):
        super(L_epe, self).__init__()
        self.norm = L2()
    def forward(self, fx,Umo,M):
        return self.norm(M*(fx-Umo))
        
class L_ms(Module):
    def __init__(self,strides = [2,8,32,64]):
        super(L_ms, self).__init__()
        self.norm = L2()
        self.strides = strides
        self.grad_lst =[(Grad(s,True),Grad(s,False)) for s in strides]
        #self.grad_and_s_lst = [(Grad(s,False),s) for s in strides] + [(Grad(s,True),s) for s in strides]  for cuda speed test use cache memory.

        
    def forward(self, fx,Umo,M):
        lossvalue = 0
        for i , grad_pair in enumerate(self.grad_lst):
            M_t , M_f = _m_resize(M,self.strides[i])
            #print(f"M_t shape {M_t} \n M_f shape{M_f}")
            grad_t , grad_f =grad_pair
            lossvalue += self.norm( M_t*(grad_t(fx)-grad_t(Umo) )) + self.norm( M_f*( grad_f(fx) - grad_f(Umo) ) )
        return lossvalue
        
class L_rec(Module):
    #flow factor should be negative
    def __init__(self,flow_factor):
        super(L_rec, self).__init__()
        self.norm = L1()
        self.flow_factor = (-1) *flow_factor 
        self.resample = Resample2d()
    def forward(self, fx, Xo,Xm):
        fx = fx*self.flow_factor
        Xo_like = self.resample(Xm,fx)
        return self.norm(Xo_like,Xo)

class L_tot(Module):
    def __init__(self,s_epe,s_ms,s_rec,flow_factor):
        super(L_tot,self).__init__()
        self.s_epe = s_epe
        self.s_ms = s_ms
        self.s_rec = s_rec
        self.loss_epe = L_epe()
        self.loss_ms = L_ms()
        self.loss_rec = L_rec(flow_factor)
    def forward(self, fx, Xo , Xm ,Umo , M):
        return self.s_epe*self.loss_epe(fx,Umo,M) # +self.s_ms*self.loss_ms(fx,Umo,M)+self.s_rec*self.loss_rec(fx,Xo,Xm)