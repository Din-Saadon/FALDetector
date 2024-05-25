import argparse
import os
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from imageio.v2 import imread
from PIL import Image
from packages.resample2d_package.resample2d import Resample2d
from modules.DRN.DRNSeg import DRNSeg
from utils.tools import *
from utils.visualize import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", required=True, help="the model input")
    parser.add_argument(
        "--dest_folder", required=True, help="folder to store the results")
    parser.add_argument(
        "--model_path", required=True, help="path to the drn model")
    parser.add_argument(
        "--gpu_id", default='0', help="the id of the gpu to run model on")
    parser.add_argument('-f', '--flow_factor', type=float, default=1.05, 
                        help='Factor used for the resample function (default: 1.05).')
    parser.add_argument('-i', '--input_size', type=str, default='800,800', 
                        help='Input size for the images (default: 800,800).')
    args = parser.parse_args()
    args.input_size = tuple(map(int, args.input_size.split(',')))
    
    img_path = args.input_path
    dest_folder = args.dest_folder
    model_path = args.model_path
    gpu_id = args.gpu_id
    flow_factor = args.flow_factor * (-1)
    input_size = args.input_size
    print(input_size)
    # Loading the model
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(gpu_id)
    else:
        device = 'cpu'

    model = DRNSeg(2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['model'])
    model.to(device)
    model.eval()
    # Data preprocessing
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    face = read_gen(img_path)
    face_tens = resize_gen(face,input_size)
    face_tens = tf(face_tens).to(device)
    print("face shape :: ",face_tens.shape)
    # Warping field prediction
    with torch.no_grad():
        flow = model(face_tens.unsqueeze(0))
        flow = flow.contiguous()
        flow_numpy = flow[0].cpu().numpy()
        flow_numpy = np.transpose(flow_numpy, (1, 2, 0))
        _,_, h, w = flow.shape
        print(flow.shape)

    # Undoing the warps
    makePIL =ToPILImage()
    modified_np =resize_gen(face,(w, h))
    modified_torch =  toTensor(modified_np).unsqueeze(0).contiguous()
    modified = makePIL(modified_torch.squeeze())
    warp= Resample2d()
    reverse_torch = warp(modified_torch, flow_factor*flow)
    reverse_torch = reverse_torch.squeeze()/255.
    reverse = makePIL(reverse_torch)
    # Saving the results
    modified.save(
        os.path.join(dest_folder, 'cropped_input.jpg'),
        quality=90)
    reverse.save(
        os.path.join(dest_folder, 'warped.jpg'),
        quality=90)
    flow_magn = np.sqrt(flow_numpy[:, :, 0]**2 + flow_numpy[:, :, 1]**2)
    save_heatmap_cv(
        modified_np, flow_magn,
        os.path.join(dest_folder, 'heatmap.jpg'))
        