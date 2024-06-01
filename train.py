
import argparse
import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from torchvision.transforms import v2
from utils.tools import SeedSetter
from modules.DRN.DRNSeg import DRNSeg
from dataset import Photoshopped_Faces
from losses import L_tot

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
    parser.add_argument('--input_size', type=str, default='256,256',
                        help='Input size for the images (default: 256,256).')
    parser.add_argument('--flow_factor', type=float, default=1.05,
                        help='Factor used for the resample function (default: 1.05).')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to the checkpoint of the DRN module (default: None).')
    parser.add_argument('-s', '--save_checkpoint', type=str, required=True,
                        help='Path where the checkpoint should be saved (default: None).')
    parser.add_argument('-epe', '--s_epe', type=float, default=1.0,
                        help='Scalar for Loss EPE (default: 1).')
    parser.add_argument('-ms', '--s_ms', type=float, default=15.0,
                        help='Scalar for multi-scale loss (default: 15).')
    parser.add_argument('-rec', '--s_rec', type=float, default=1.5,
                        help='Scalar for reconstruction loss (default: 1.5).')
    parser.add_argument('-strides', '--strides_ms', type=str, default='2,8,32,64',
                        help='Strides for multi-scale loss (default: 2,8,32,64).')
    parser.add_argument('-epochs','--num_epochs', type=int, default=1,
                        help='Number of epochs (default: 1).')
    parser.add_argument('-bs','--batch_size', type=int, default=32,
                        help='Batch size (default: 32).')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4).')
    parser.add_argument('--b1', type=float, default=0.9,
                        help='Beta1 for Adam optimizer (default: 0.9).')
    parser.add_argument('--b2', type=float, default=0.999,
                        help='Beta2 for Adam optimizer (default: 0.999).')

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
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    b1 = args.b1
    b2 = args.b2
       # Print the arguments
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
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Beta1 for Adam optimizer: {b1}")
    print(f"Beta2 for Adam optimizer: {b2}")

    return (modified_img_dir, original_img_dir, input_size, flow_factor, 
            checkpoint, save_checkpoint, s_epe, s_ms, s_rec, strides_ms, 
            num_epochs, batch_size, lr, b1, b2)

if __name__ == "__main__":
    (modified_img_dir, original_img_dir, input_size, flow_factor, 
     checkpoint, save_checkpoint, s_epe, s_ms, s_rec, strides_ms, 
     num_epochs, batch_size, lr, b1, b2) = parse_args()
    print(parse_args())
    dir_lst = [original_img_dir,modified_img_dir]
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ds = Photoshopped_Faces(dir_lst,size = input_size,transform=normalize)
    ds_size = len(ds)
    test_size = int(0.05*ds_size)
    valid_size = int(0.15*ds_size)
    train_size = ds_size - valid_size - test_size
    print("ds_size" , ds_size)
    with SeedSetter(42): 
        train ,val ,test= torch.utils.data.random_split(ds, [train_size, valid_size,test_size])
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
    #check if cuda is available:
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(0)
    else:
        device = 'cpu'
        
    #pretrained DRN model:
    model = DRNSeg(2)
    model.cuda()
    if checkpoint:
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict['model'])
    
    #set optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr,(b1,b2))
    loss_total = L_tot(s_epe = s_epe , s_ms = s_ms , s_rec = s_rec , flow_factor=flow_factor,strides = strides_ms)
    
    best_val_loss = float('inf')
    best_model_path = save_checkpoint
import torch
from torch.autograd import Variable
from tqdm import tqdm

# Assuming you have defined your model, loss function, optimizer, and data loaders

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    model.enable_input_require_grads()
    train_loss = 0.0

    # Training loop with tqdm progress bar
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False):
        batch = [t.cuda() for t in batch]  # Move data to GPU if available
        Xo, Xm, Umo, M = batch
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(Xm).to(device=device)  # Forward pass
        
        loss = loss_total(outputs, Xo, Xm, Umo, M)  
        loss.requires_grad_()# Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        train_loss += loss.item()  # Accumulate loss

    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False):
            batch = [t.cuda() for t in batch]
            Xo, Xm, Umo, M = batch
            
            outputs = model(Xm).to(device=device)  # Forward pass
            loss = loss_total(outputs, Xo, Xm, Umo, M)  # Compute loss
            val_loss += loss.item()  # Accumulate loss

    # Calculate average losses
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)  # Save best model checkpoint
        print(f'Saved best model with loss: {best_val_loss:.4f}')

    # Print epoch summary
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
