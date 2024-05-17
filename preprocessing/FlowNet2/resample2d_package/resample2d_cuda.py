import torch
from math import floor ,ceil

#To do: convert to cuda version
#input1 : [1,3,384,384] img
#input2 : [1,2,384,384] flow
#output : [1,3,384,384] warped_img
#flow[:,0,:,:] - flow in X axis
#flow[:,1,:,:] - flow in Y axis

def forward(input1, input2, output, kernel_size, bilinear):
        B,C,H,W = input2.shape
        print(f'B:{B} C:{C} H:{H} W:{W}')
        for b in range(B):
            for x in range(H):
                for y in range(W):
                    u_x = input2[b][0][x][y]
                    v_y = input2[b][1][x][y]
                    for i in range(3):
                    new_x = x+u_x
                    new_y = y+v_y
                    print(f'w(x) + x = ({new_x},{new_y})')
                    is_x_in_range = 0 <= new_x and new_x <H
                    is_y_in_range = 0 <= new_y and new_y <W
                    output[b][i][x][y]=0
                    if is_x_in_range and is_y_in_range:
                        flx = floor(new_x)
                        fly = floor(new_y)
                        cex = ceil(new_x)
                        cey = ceil(new_y)
                        Qx = new_x - flx
                        Qy = new_y - fly
                        not_Qx = 1-Qx
                        not_Qy = 1-Qy
                        print(f'Qx={Qx} ; Qy={Qy}')
                        output[b][i][x][y] = not_Qx*not_Qy*input1[b][i][flx][fly]+\
                        not_Qx*Qy*input1[b][i][flx][cey] + \
                        Qx*not_Qy*input1[b][i][cex][fly] + \
                        Qx*Qy*input1[b][i][cex][cey]
                        

