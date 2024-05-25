import time
import torch
import os
import numpy as np
from .flow_utils import readFlow , flow_resize
from imageio import imread
import cv2
from PIL import Image

def resize_shorter_side(img, min_length):
    """
    Resize the shorter side of img to min_length while
    preserving the aspect ratio.
    """
    ow, oh = img.size
    mult = 8
    if ow < oh:
        if ow == min_length and oh % mult == 0:
            return img, (ow, oh)
        w = min_length
        h = int(min_length * oh / ow)
    else:
        if oh == min_length and ow % mult == 0:
            return img, (ow, oh)
        h = min_length
        w = int(min_length * ow / oh)
    return img.resize((w, h), Image.BICUBIC), (w, h)

def grouped_files_from_dir_list(dir_lst):
    files = [sorted([os.path.join(directory,file) for file in os.listdir(directory)]) for directory in dir_lst]
    return list(zip (*files) )

def toTensor(x_numpy,type =torch.float):
    res = torch.tensor(x_numpy,dtype = type)
    return res.permute(2,0,1).cuda()

class SeedSetter:
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_state)

class TimerBlock: 
    def __init__(self, title):
        print(("{}".format(title)))

    def __enter__(self):
        self.start = time.process_time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.process_time()
        self.interval = self.end - self.start

        if exc_type is not None:
            self.log("Operation failed\n")
        else:
            self.log("Operation finished\n")


    def log(self, string):
        duration = time.process_time() - self.start
        units = 's'
        if duration > 60:
            duration = duration / 60.
            units = 'm'
        print(("  [{:.3f}{}] {}".format(duration, units, string)))
    
    def log2file(self, fid, string):
        fid = open(fid, 'a')
        fid.write("%s\n"%(string))
        fid.close()

def resize_gen(x_numpy,size):
    if(x_numpy.shape[2] == 2): # flow
        return flow_resize(x_numpy,size)
    assert(x_numpy.shape[2]==3) # image
    return cv2.resize(x_numpy, size, interpolation=cv2.INTER_CUBIC)

def read_gen(file_name):
    ext = file_name.split('.')[-1]
    if ext == 'png' or ext == 'jpeg' or ext == 'ppm' or ext == 'jpg':
        im = imread(file_name)
        if im.shape[2] > 3:
            return im[:,:,:3]
        else:
            return im
    elif ext == 'bin' or ext == 'raw':
        return np.load(file_name)
    elif ext == 'flo':
        return readFlow(file_name).astype(np.float32)
    return []