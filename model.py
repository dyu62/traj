import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import Linear, ModuleList, ReLU
import torch.nn.functional as F
from torch.utils.data import Dataset
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.layers = ModuleList()
        input_fc = nn.Linear(input_dim, hidden_dim, bias=False)
        output_fc = nn.Linear(hidden_dim, output_dim, bias=False)
        self.layers.append(input_fc)
        if num_layers > 2:
            fc = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.layers.append(fc)
        self.layers.append(output_fc)
        self.num_layers = num_layers

    def forward(self, x):
        for i in range(self.num_layers-1):
            layer = self.layers[i]
            x = layer(x)
            x = F.relu(x)
        layer = self.layers[-1]
        x = layer(x)
        return x
    
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.layers = ModuleList()
        rnn = torch.nn.GRU(input_dim,hidden_dim,num_layers,bias=False)
        output_fc = nn.Linear(hidden_dim, output_dim, bias=False)
        self.layers.append(rnn)
        self.layers.append(output_fc)
        self.num_modules = 2
    def forward(self, x):
        for i in range(self.num_modules-1):
            layer = self.layers[i]
            x = layer(x)
            if type(x) is tuple:
                x=x[0][-1]
            x = F.relu(x)
        layer = self.layers[-1]
        x = layer(x)
        return x
class Transformer(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.layers = ModuleList()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        output_fc = nn.Linear(hidden_dim, output_dim, bias=False)
        self.layers.append(transformer_encoder)
        self.layers.append(output_fc)
        self.num_modules = 2
    def forward(self, x):
        for i in range(self.num_modules-1):
            layer = self.layers[i]
            x = layer(x)
            if type(x) is tuple:
                x=x[0][-1]
            x = F.relu(x)
        layer = self.layers[-1]
        x = layer(x)
        return x   
class OneClassModel(nn.Module):
    def __init__(self, input_dim,input_dim_SRNN,input_dim_TRNN, hidden_dim, output_dim, num_layers=2, r=.1):
        super().__init__()
        self.encoder = MLP(input_dim, hidden_dim, output_dim, num_layers)
        self.Sencoder = RNN(input_dim_SRNN, hidden_dim, int(output_dim/2), num_layers)
        self.Tencoder = RNN(input_dim_TRNN, hidden_dim, int(output_dim/2), num_layers)
        self.Sencoder_Transformer = Transformer(hidden_dim, int(output_dim/2), num_layers)
        self.Tencoder_Transformer = Transformer(hidden_dim, int(output_dim/2), num_layers)        
        self.center = torch.ones(output_dim, dtype=torch.float,device="cuda")
        self.r = torch.nn.Parameter(data=torch.tensor(r,dtype=torch.float), requires_grad=True)
    def forward(self, x):
        return self.encoder(x)
    def forward_RNN(self,x_s,x_t):
        embeds_s = self.Sencoder(x_s)
        embeds_t = self.Tencoder(x_t)
        embeds=torch.cat([embeds_s,embeds_t],1)
        return embeds
    def forward_Transformer(self,x_s,x_t):
        embeds_s = self.Sencoder(x_s)
        embeds_t = self.Tencoder(x_t)
        embeds=torch.cat([embeds_s,embeds_t],1)
        return embeds    
    def score(self, x):
        embeds = self.encoder(x)
        score = torch.cdist(embeds, self.center.view(1,-1))
        return score
    def score_RNN(self, x_s,x_t):
        embeds_s = self.Sencoder(x_s)
        embeds_t = self.Tencoder(x_t)
        embeds=torch.cat([embeds_s,embeds_t],1)
        score = torch.cdist(embeds, self.center.view(1,-1))
        return score
    def score_Transformer(self, x_s,x_t):
        return self.score_RNN(x_s,x_t)     
    def oneclassLoss(self, x):
        embeds = self.encoder(x)
        distances = torch.cdist(embeds, self.center.view(1,-1))
        return torch.mean(torch.relu(distances - self.r))
    def oneclassLoss_RNN(self, x_s,x_t):
        embeds_s = self.Sencoder(x_s)
        embeds_t = self.Tencoder(x_t)
        embeds=torch.cat([embeds_s,embeds_t],1)
        distances = torch.cdist(embeds, self.center.view(1,-1))
        return torch.mean(torch.relu(distances - self.r))
    def oneclassLoss_Transformer(self, x_s,x_t):
        embeds_s = self.Sencoder(x_s)
        embeds_t = self.Tencoder(x_t)
        embeds=torch.cat([embeds_s,embeds_t],1)
        distances = torch.cdist(embeds, self.center.view(1,-1))
        return torch.mean(torch.relu(distances - self.r))
