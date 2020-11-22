import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ProteinNetDataset
from model import ResNet
import time
import logging

df = './input/small_dataset.csv'
pssm_dir = 'pssm'
tert_dir = 'tert'
max_len = 250
batch_size = 1
input_shape = (1, 84, 250, 250)
n_dist_bins = 10
n_blocks = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = ProteinNetDataset(df, pssm_dir, tert_dir, max_len)
loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # false for testing

model = ResNet(input_shape, n_dist_bins, n_blocks).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    epoch_loss = 0
    epoch_acc = 0

    for batch_idx, (feats, dist_arr) in enumerate(loader):

        feats = feats.to(device).float()
        dist_arr = dist_arr.to(device).long().squeeze(0)  # remove batch dim for loss
        elems = (dist_arr != -1).sum()  # non-padding elements

        # forward
        out = model(feats)
        loss = criterion(out, dist_arr)

        # update grads and backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # stats
        pred = torch.argmax(out, dim=1)
        correct = (pred == dist_arr).sum().float()
        acc = correct / elems

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    # normalize per batch
    return epoch_loss / len(loader), epoch_acc / len(loader)

for i in range(10):
    loss, acc = train_one_epoch(model, loader, optimizer, criterion)
    print(i)
    print(loss, acc)

