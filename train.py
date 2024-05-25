import argparse
# lr , Adam : b1 , b2 num_epochs
def parse_args():
    parser = argparse.ArgumentParser(
        description="Process some arguments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-p', '--preprocessing', action='store_true', default=False, 
                        help='Flag to indicate if preprocessing should be done.')
    
    parser.add_argument('-Xm', '--modified_img_dir', type=str, required=True, 
                        help='Directory containing modified images.')
    
    parser.add_argument('-Xo', '--original_img_dir', type=str, required=True, 
                        help='Directory containing original images.')
    
    parser.add_argument('-Umo', '--Umo_dir', type=str, required=True, 
                        help='Directory containing optical flow from modified to original images.')
    
    parser.add_argument('-Uom', '--Uom_dir', type=str, required=True, 
                        help='Directory containing optical flow from original to modified images.')
    
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

    args = parser.parse_args()

    # Convert comma-separated strings to tuples or lists as needed
    args.input_size = tuple(map(int, args.input_size.split(',')))
    args.strides_ms = list(map(int, args.strides_ms.split(',')))

    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)
