import time
import torch
import os
import numpy as np
 
def grouped_files_from_dir_list(dir_lst):
    files = [sorted([os.path.join(directory,file) for file in os.listdir(directory)]) for directory in dir_lst]
    return list(zip (*files) )

def toTensor(x_numpy,type =torch.float):
    res = torch.tensor(x_numpy,dtype = type)
    return res.permute(2,0,1).unsqueeze(0).cuda()

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
