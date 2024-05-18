import torch
import numpy as np
import argparse

from models import FlowNet2
from datasets import ImagesFromFolder
from utils.frame_utils import read_gen  # the path is depended on where you create this module

if __name__ == '__main__':
    # obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    parser.add_argument('--crop_size', type=int, nargs='+', default = [256, 256], help="Spatial dimension to crop training samples for training")
    parser.add_argument('--inference_size', type=int, nargs='+', default = [-1,-1], help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
    
    args = parser.parse_args()

    # initial a Net
    net = FlowNet2(args).cuda()
    # load the state_dict
    print("load")
    dict = torch.load("./FlowNet2_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"])
    ds = ImagesFromFolder(args,False,root="./demo")
    for i, (x,y) in enumerate(ds):
        print(x[0].shape)
        if i==5:
            break
    print("predict")
    # process the image pair to obtian the flow
    #result = net(im).squeeze()


    # save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project
    def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()

    print("save")
    #data = result.data.cpu().numpy().transpose(1, 2, 0)
    #writeFlow("/demo/res/flow.flo", data)
