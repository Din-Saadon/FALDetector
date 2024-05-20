from utils.geometry import forward_backward_consistency_check
from utils.tools import grouped_files_from_dir_list,SeedSetter,toTensor
from utils.flow_utils import readFlow
import os
import numpy as np
import torch
from torchvision.transforms import v2

#paths:

sources = "../sources"
images_dir = sources+"/images"
flow_res = sources+"/flows"
m_dir = flow_res + "/M"
flow_res_fwd = flow_res + "/fwd"
flow_res_bwd = flow_res + "/bwd"
mod = images_dir + "/modified"
ref = images_dir + "/reference"
Xm1 = mod + "/openimage_0995.png"
Xo1 = ref + "/openimage_0995.png"
res_example = images_dir + "/res_example"
flownet = flow_res + "/FlowNet2.flo"
spynet = flow_res + "/openimage_0995.flo"
openimage_0995_res = res_example + "/openimage_0995_ref_like.png"
checkpoints = "../../checkpoints"
flownet2_checkpoint = checkpoints + "/FlowNet2/FlowNet2_checkpoint.pth.tar"


#../sources/images/modified ../sources/images/reference
#../sources/flows/bwd ../sources/flows/fwd
os.system("python modules/SpyNet/pred.py --model sintel-final --one ../sources/images/modified --two ../sources/images/reference --out ../sources/flows/fwd")
os.system("python modules/SpyNet/pred.py --model sintel-final --one ../sources/images/reference --two ../sources/images/modified --out ../sources/flows/bwd ")

seed_value = 42
G = v2.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 5.))

path_lst = grouped_files_from_dir_list([flow_res_fwd,flow_res_bwd])
for pair in path_lst:
    fwd_name , bwd_name = pair
    fwd = toTensor(readFlow(fwd_name))
    bwd = toTensor(readFlow(bwd_name))
    M_inconsistent , _ = forward_backward_consistency_check(fwd,bwd,alpha=0.85,beta=0.085)
    with SeedSetter(seed_value):
        M = 1 - G(M_inconsistent)
    #TO DO: save M in source dir
    name_img = fwd_name[:-4]
    name_img = name_img.split('/')[-1]
    flow_path = m_dir + "/" + name_img + ".pt"
    torch.save(M, flow_path)