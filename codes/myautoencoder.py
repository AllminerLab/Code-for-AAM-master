#from preprocess import readdata
import torch
from torch import nn

class myAutoencoder(nn.Module):
    def __init__(self, input_x):
        super(myAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_x, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_x),
            nn.ReLU(),       
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


















