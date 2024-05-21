from utils.geometry import forward_backward_consistency_check
from utils.tools import grouped_files_from_dir_list,SeedSetter,toTensor
from utils.tools import read_gen , resize_gen
from torch.utils.data import Dataset
from torchvision.transforms import v2


class MyDataset(Dataset):
    def __init__(self, dir_lst ,transform=None , size = (800,800) , alpha=0.85,beta=0.085, G = v2.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 5.))):
        assert(len(dir_lst) == 4) # Xo , Xm ,Umo , Uom.
        self.dir_lst = dir_lst 
        self.transform = transform
        self.size =size
        self.path_lst = grouped_files_from_dir_list(dir_lst)
        self.alpha = alpha
        self.beta =beta
        self.G = G

    def __len__(self):
        return len(self.path_lst)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        outputs = [toTensor(resize_gen(read_gen(path),self.size)) for path in self.path_lst[idx]]
        Xo ,Xm ,Umo ,Uom = outputs
        if self.transform:
            Xo = self.transform(Xo)
            Xm = self.transform(Xm)
        #flow consistency map:
        M_inconsistent , _ = forward_backward_consistency_check(Umo.unsqueeze(0),Uom.unsqueeze(0),self.alpha,self.beta)
        M = 1 - self.G(M_inconsistent)
        
        return (Xo,Xm,Umo,M)