import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.convolutional_model = nn.Sequential(
            nn.ConvTranspose2d(8,32,(3,3),padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,128,(3,3),padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,128,(4,4),stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,(4,4),stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,16,(4,4),stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,3,(3,3),padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.reshape(x,(x.shape[0],x.shape[1]//4//4,4,4))
        x = self.convolutional_model(x)
        return x

# model = Decoder().to('cpu')
# print(model)