import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import Linear, ModuleList, ReLU
import torch.nn.functional as F
from torch.utils.data import Dataset
class myDataset(Dataset):
    def __init__(self, data):
        # with open(os.path.join(data_dir,file+"_ri.npy"), 'rb') as f:
        #     data = np.load(f)
        self.dataS=data[0]
        self.dataT=data[1] 
    def __len__(self) -> int:   #可以sample的数量
        return self.dataS.shape[0]
    def __getitem__(self, index):   #iNdex global
        return self.dataS[index],self.dataT[index]