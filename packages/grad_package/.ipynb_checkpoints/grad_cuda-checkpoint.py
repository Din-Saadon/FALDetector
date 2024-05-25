import torch
# TO DO: convert to cuda package
class grad_cuda():
    @staticmethod
    #t = True => x axis , t=False => y axis
    def forward(input1, output, s, t):
      B,D,H,W = input1.size()
      sx,sy  = (0,s) if t else (s,0)
      for sample in range(B):
        for channel in range(D):
          for i in range(H):
            for j in range(W):
              a_i , a_j = i-sx, j-sy
              b_i,b_j =i+sx,j+sy
              in_range = a_i>=0 and a_j>=0 and b_i<H and b_j<W
              if in_range:
                output[sample,channel,a_i,a_j] = (input1[sample,channel,b_i,b_j]-input1[sample,channel,a_i,a_j])/2*s
      return

    @staticmethod
    def backward(grad_input1,s, t):
      sx = 0 if t else 2*s
      sy = 0 if not t else 2*s
      b,d,h_out,w_out,h,w = grad_input1.size()
      for sample in range(b):
        for channel in range(d):
          for i in range(h_out):
            for j in range(w_out):
              grad_input1[sample,channel,i,j,i,j] = -1/2*s
              grad_input1[sample,channel,i,j,i+sx,j+sy] = 1/2*s



      return

