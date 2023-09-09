import torch
from torch import nn
from torch.autograd import Variable



class myMLP(nn.Module):
    def __init__(self):
        super(myMLP,self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
    def forward(self,input_ae):
        dout = self.mlp(input_ae)
        return dout






