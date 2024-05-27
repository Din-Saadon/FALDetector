"""
import argparse
import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function, Variable
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.transforms import v2
from utils.geometry import forward_backward_consistency_check
from utils.tools import (
    grouped_files_from_dir_list,
    SeedSetter,
    toTensor,
    read_gen,
    resize_gen,
    TimerBlock
)
from modules.DRN.DRNSeg import DRNSeg
from dataset import Photoshopped_Faces
from losses import L_tot
"""
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process some arguments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-Xm', '--modified_img_dir', type=str, required=True,
                        help='Directory containing modified images.')
    parser.add_argument('-Xo', '--original_img_dir', type=str, required=True,
                        help='Directory containing original images.')
    parser.add_argument('-i', '--input_size', type=str, default='800,800',
                        help='Input size for the images (default: 800,800).')
    parser.add_argument('-f', '--flow_factor', type=float, default=1.05,
                        help='Factor used for the resample function (default: 1.05).')
    parser.add_argument('-c', '--checkpoint', type=str, default=None,
                        help='Path to the checkpoint of the DRN module (default: None).')
    parser.add_argument('-s', '--save_checkpoint', type=str, default=None,
                        help='Path where the checkpoint should be saved (default: None).')
    parser.add_argument('-epe', '--s_epe', type=float, default=1.0,
                        help='Scalar for Loss EPE (default: 1).')
    parser.add_argument('-ms', '--s_ms', type=float, default=15.0,
                        help='Scalar for multi-scale loss (default: 15).')
    parser.add_argument('-rec', '--s_rec', type=float, default=1.5,
                        help='Scalar for reconstruction loss (default: 1.5).')
    parser.add_argument('-strides', '--strides_ms', type=str, default='2,8,32,64',
                        help='Strides for multi-scale loss (default: 2,8,32,64).')

    # Adding num_of_epochs and batch_size arguments
    parser.add_argument('--num_of_epochs', type=int, default=20,
                        help='Number of epochs (default: 20).')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64).')

    args = parser.parse_args()

    # Convert comma-separated strings to tuples or lists as needed
    args.input_size = tuple(map(int, args.input_size.split(',')))
    args.strides_ms = list(map(int, args.strides_ms.split(',')))

    # Assigning parsed values to variables
    modified_img_dir = args.modified_img_dir
    original_img_dir = args.original_img_dir
    input_size = args.input_size
    flow_factor = args.flow_factor
    checkpoint = args.checkpoint
    save_checkpoint = args.save_checkpoint
    s_epe = args.s_epe
    s_ms = args.s_ms
    s_rec = args.s_rec
    strides_ms = args.strides_ms
    num_of_epochs = args.num_of_epochs
    batch_size = args.batch_size

    return (modified_img_dir, original_img_dir, input_size, flow_factor, 
            checkpoint, save_checkpoint, s_epe, s_ms, s_rec, strides_ms, 
            num_of_epochs, batch_size)

if __name__ == "__main__":
    (modified_img_dir, original_img_dir, input_size, flow_factor, 
     checkpoint, save_checkpoint, s_epe, s_ms, s_rec, strides_ms, 
     num_of_epochs, batch_size) = parse_args()
    
    print(f"Modified image directory: {modified_img_dir}")
    print(f"Original image directory: {original_img_dir}")
    print(f"Input size: {input_size}")
    print(f"Flow factor: {flow_factor}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Save checkpoint: {save_checkpoint}")
    print(f"Scalar for Loss EPE: {s_epe}")
    print(f"Scalar for multi-scale loss: {s_ms}")
    print(f"Scalar for reconstruction loss: {s_rec}")
    print(f"Strides for multi-scale loss: {strides_ms}")
    print(f"Number of epochs: {num_of_epochs}")
    print(f"Batch size: {batch_size}")
"""
        
    dir_lst = [ref,mod]
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ds = Photoshopped_Faces(dir_lst,size = input_size,transform=normalize)
    train ,val ,test= torch.utils.data.random_split(ds, [800, 180,20])
    train_loader = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
"""
