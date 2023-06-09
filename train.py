import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import Linear, ModuleList, ReLU
import torch.nn.functional as F
from torch.utils.data import Dataset
import model
import dataset


import argparse
import os
import random
import torch
import pandas as pd
import numpy as np
import time
import torch.optim as optim
import scipy
from matplotlib import cm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.functional import softmax
import copy

torch.autograd.set_detect_anomaly(True)
from sklearn.metrics import explained_variance_score,mean_squared_error,mean_absolute_error,r2_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve, auc
from sklearn.feature_selection import r_regression
import pickle
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets
import torchvision.models
import math
import shutil
import time
blue = lambda x: '\033[94m' + x + '\033[0m'
red = lambda x: '\033[31m' + x + '\033[0m'
green = lambda x: '\033[32m' + x + '\033[0m'
yellow = lambda x: '\033[33m' + x + '\033[0m'
greenline = lambda x: '\033[42m' + x + '\033[0m'
yellowline = lambda x: '\033[43m' + x + '\033[0m'

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', default='data/allsolar_png512', type=str)
    parser.add_argument('--data_dir', default='data/allsolar_full_png512', type=str)
    
    parser.add_argument('--model',default="our", type=str)
    parser.add_argument('--train_batch', default=128, type=int)
    parser.add_argument('--test_batch', default=16, type=int)
    
    parser.add_argument('--h_ch', default=32, type=int)
    parser.add_argument('--eta', default=0.01, type=float) #tolerant
    
    parser.add_argument('--dataset', type=str, default='traj')
    parser.add_argument('--log', type=str, default="True")
    parser.add_argument('--loadmodel', type=str, default="False")     
    parser.add_argument('--test_per_round', type=int, default=10)
    parser.add_argument('--patience', type=int, default=30)  #scheduler
    parser.add_argument('--nepoch', type=int, default=201)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--manualSeed', type=str, default="False")
    parser.add_argument('--man_seed', type=int, default=12345)
    args = parser.parse_args()
    args.log=True if args.log=="True" else False
    args.loadmodel=True if args.loadmodel=="True" else False    
    args.save_dir=os.path.join('./save/',args.dataset)
    args.manualSeed=True if args.manualSeed=="True" else False
    return args
def main(args,train_Loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mymodel = model.OneClassModel(input_dim=51*2,input_dim_SRNN=2,input_dim_TRNN=2, hidden_dim=16, output_dim=4, num_layers=3)
    mymodel.to(device)
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=args.patience, min_lr=1e-8)   

    def train(model):
        totaltime=0
        epochloss=0
        y_hat, y_true,y_hat_logit = [], [], []        
        optimizer.zero_grad()
        model.train()
        for batch_idx, (dataS,dataT) in enumerate(train_Loader):
            batchsize=dataS.shape[0]
            dataS, dataT = dataS.permute(1,0,2),dataT.permute(1,0,2) #length batch size
            dataS, dataT = dataS.to(device), dataT.to(device)
            
            loss = mymodel.oneclassLoss_RNN(dataS, dataT) + mymodel.r**2
            loss = mymodel.oneclassLoss_Transformer(dataS, dataT) + mymodel.r**2
            
            optimizer.zero_grad()
            loss.backward()
            epochloss+=loss.detach()
            optimizer.step()
        print("Epoch:{:03d}, loss={:.4f}, r={:.4f}".format(epoch, float(epochloss), float(mymodel.r)))
        return epochloss.item()/len(train_Loader)
    
    for epoch in range(args.nepoch):
        train_loss=train(mymodel)
    suffix="{}{}-{}:{}:{}".format(datetime.now().strftime("%h"),
                                    datetime.now().strftime("%d"),
                                    datetime.now().strftime("%H"),
                                    datetime.now().strftime("%M"),
                                    datetime.now().strftime("%S"))          
    torch.save(mymodel.state_dict(),os.path.join(args.save_dir,'model'+"_"+suffix+'.pth'))
    
if __name__=='__main__':
    args = get_args()
    data = torch.tensor(np.load("train.npy"), dtype=torch.float)
    data_T=torch.rand(data.shape)   
    ds=dataset.myDataset([data,data_T]) 
    train_loader=torch.utils.data.DataLoader(ds,batch_size=args.train_batch, shuffle=True) 
    main(args,train_loader)