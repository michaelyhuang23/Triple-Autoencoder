import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.convolutional_model = nn.Sequential(
            nn.Conv2d(3,64 ,(3,3),padding=1),
            nn.Conv2d(64,128,(3,3),padding=1),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(128,128,(3,3),padding=1),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(128,64,(3,3),padding=1),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64,32,(3,3),padding=1)
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(512,128)

    def forward(self, x):
        x = self.convolutional_model(x)
        x = self.flatten(x)
        x = self.linear1(x)
        return x

# model = Encoder().to('cpu')
# print(model)