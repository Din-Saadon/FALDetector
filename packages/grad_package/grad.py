
from torch.nn.modules.module import Module
from torch.autograd import Function, Variable

class GradFunction(Function):
    @staticmethod
    def forward(ctx, input1, s=1, t= True):
        assert input1.is_contiguous()

        ctx.save_for_backward(input1)
        ctx.s = s
        ctx.t = t
        b, d, h, w = input1.size()
        if t:
          w=w-2*s
        else:
          h=h-2*s
        output = input1.new(b, d, h, w).zero_()

        grad_cuda().forward(input1, output, s,t)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        assert grad_output.is_contiguous()

        input1 = ctx.saved_tensors
        b, d, h, w = input1.size()
        _, _ , h_out, w_out = grad_output.size()
        grad_input1 = Variable(input1.new_zeros(b,d,h_out,w_out,h,w))

        grad_cuda().backward(grad_input1.data,ctx.s, ctx.t)
        print(grad_input1)

        return grad_input1, None, None

class Grad(Module):

    def __init__(self, s=1, t = True):
        super(Grad, self).__init__()
        self.s = s
        self.t = t

    def forward(self, input1):
        input1_c = input1.contiguous()
        return GradFunction.apply(input1_c , self.s, self.t)
