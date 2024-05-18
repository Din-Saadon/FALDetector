import numpy as np
import argparse
import torch
from torch.autograd import Variable

from torch.utils.data import DataLoader
from tqdm import tqdm

from networks.resample2d_package.resample2d import Resample2d
from datasets import ImagesFromFolder
from datasets import StaticRandomCrop , StaticCenterCrop
import utils.frame_utils as frame_utils


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape testdata into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))
if __name__ == '__main__':
    # obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    parser.add_argument('--crop_size', type=int, nargs='+', default = [256, 256], help="Spatial dimension to crop training samples for training")
    parser.add_argument('--inference_size', type=int, nargs='+', default = [-1,-1], help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
    args = parser.parse_args()


    ds = ImagesFromFolder(args , is_cropped = False , root = "./demo")
    data_loader = DataLoader(ds, batch_size=4, shuffle=False)
    progress = tqdm(data_loader, ncols=100, total=np.minimum(len(data_loader), 4), desc='Inferencing ', leave=True, position=0)
    for batch_idx, (data, target) in enumerate(progress):
        print()
        if torch.cuda.is_available():
            data, target = [d.cuda(non_blocking=True) for d in data], [t.cuda(non_blocking=True) for t in target]
        data, target = [Variable(d) for d in data], [Variable(t) for t in target]
        
        with torch.no_grad():
            inputs  = data[0]
            flow = target[0]
            rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
            x = (inputs - rgb_mean) / args.rgb_max
            x1 = x[:,:,0,:,:]
            x2 = x[:,:,1,:,:]
            x = torch.cat((x1,x2), dim = 1)
            resample = Resample2d()
            resampled_img1 = resample(x[:,3:,:,:],flow )
            print(resampled_img1.shape)


#    img1_like_torch  = Resample2d(img0_numpy,flow_numpy)  ## warp via optical flow img1 to img
#    img1_like = Image.fromarray(img1_like_numpy)
#    img1_like_path = "." + "/openimage_0995_1_like.png"
#    img1_like.save(img1_like_path)
#    diff_show(img1_like_path,img1_path)

"""
rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))

x = (inputs - rgb_mean) / self.rgb_max
x1 = x[:,:,0,:,:]
x2 = x[:,:,1,:,:]
x = torch.cat((x1,x2), dim = 1)

# flownetc
flownetc_flow2 = self.flownetc(x)[0]
flownetc_flow = self.upsample1(flownetc_flow2*self.div_flow)

# warp img1 to img0; magnitude of diff between img0 and and warped_img1, 
resampled_img1 = self.resample1(x[:,3:,:,:], flownetc_flow)
diff_img0 = x[:,:3,:,:] - resampled_img1 
norm_diff_img0 = self.channelnorm(diff_img0)
"""