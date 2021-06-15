import torch
from torch import nn
from decoder import Decoder
from encoder import Encoder

class CompModel(nn.Module):
    def __init__(self):
        super(CompModel, self).__init__()
        self.encoder = Encoder()
        self.classifier = nn.Sequential(
            nn.Linear(128,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,100)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
