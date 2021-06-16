import torch
from torch import nn
from decoder import Decoder
from encoder import Encoder

class TripleAutoencoder(nn.Module):
    def __init__(self):
        super(TripleAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.recognizer = Encoder()
        self.classifier = nn.Sequential(
            nn.Linear(128,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,100)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.recognizer(x)
        x = self.classifier(x)
        return x
