import torch
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from compmodel import CompModel
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data = DataLoader(datasets.CIFAR100(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
),batch_size=64)

test_data = DataLoader(datasets.CIFAR100(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
),batch_size=64)

model = CompModel().to(device)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        st = time.time()
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        correct+=(torch.argmax(pred,-1)==y).float().sum()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss, current = loss.item(), batch * len(X)
        sys.stdout.write("training: %d/%d;  train loss is: %f;  train acc is: %f;   time per batch: %f \r" %
                         (current, size, loss, correct/current, time.time()-st))
        sys.stdout.flush()
    print(f"final train acc is {correct/size}")

def eval(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        st = time.time()
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        correct+=(torch.argmax(pred,-1)==y).float().sum()
        loss, current = loss.item(), batch * len(X)
        sys.stdout.write("evaluating: %d/%d;  eval loss is: %f;  eval acc is: %f;  time per batch: %f \r" %
                         (current, size, loss, correct/current, time.time()-st))
        sys.stdout.flush()
    print(f"final eval acc is {correct/size}")

loss_fn = CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.00003)


EPOCH = 5

for epoch in range(EPOCH):
    print(f"training epoch {epoch}")
    train(train_data,model,loss_fn,optimizer)
    with torch.no_grad():
        print(f"evaluating epoch {epoch}")
        eval(test_data,model,loss_fn)
