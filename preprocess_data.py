import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description="Parse image directory paths and flow directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Argument for modified image directory path
    parser.add_argument('--mod', type=str, default='../sources/images/modified',
                        help='Path to the directory containing modified images.')

    # Argument for reference image directory path
    parser.add_argument('--ref', type=str, default='../sources/images/reference',
                        help='Path to the directory containing reference images.')

    # Argument for forward flow directory path
    parser.add_argument('--flow_fwd', type=str, default='../sources/flows/fwd',
                        help='Path to the directory where forward flow files will be saved.')

    # Argument for backward flow directory path
    parser.add_argument('--flow_bwd', type=str, default='../sources/flows/bwd',
                        help='Path to the directory where backward flow files will be saved.')

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments and update paths in the os.system calls
    mod_path = args.mod
    ref_path = args.ref
    flow_fwd_path = args.flow_fwd
    flow_bwd_path = args.flow_bwd

    # Ensure the output directories exist
    os.makedirs(flow_fwd_path, exist_ok=True)
    os.makedirs(flow_bwd_path, exist_ok=True)

    # pred SpyNet
    os.system(f"python modules/SpyNet/pred.py --model sintel-final --one {mod_path} --two {ref_path} --out {flow_fwd_path}")
    os.system(f"python modules/SpyNet/pred.py --model sintel-final --one {ref_path} --two {mod_path} --out {flow_bwd_path}")

if __name__ == "__main__":
    main()
