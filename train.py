import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ProteinNetDataset
from model import ResNet
from collections import defaultdict
import time
import logging
from pprint import pformat
from utils import parse_params, init_logger, save_checkpoint, load_checkpoint
import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='config path')

    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    epoch_loss = 0
    epoch_acc = 0

    for batch_idx, (feats, dist_arr, seq_len, name) in enumerate(loader):

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


def run_eval(model, loader, criterion):

    # same as training without optimizer updates

    model.eval()

    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():

        for batch_idx, (feats, dist_arr, seq_len, name) in enumerate(loader):

            feats = feats.to(device).float()
            dist_arr = dist_arr.to(device).long().squeeze(0)
            elems = (dist_arr != -1).sum()

            out = model(feats)
            loss = criterion(out, dist_arr)

            pred = torch.argmax(out, dim=1)
            correct = (pred == dist_arr).sum().float()
            acc = correct / elems

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)


def run_n_epochs(epochs, model, train_loader, valid_loader, optimizer, criterion, name):

    # hold accuracy/loss at end of each epoch
    history = defaultdict(list)

    best_valid_loss = float('inf')
    start = time.time()

    for epoch in range(1, epochs+1):

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc = run_eval(model, valid_loader, criterion)
        end = time.time()

        epoch_time = (end - start)/60

        # metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)

        # save model when loss decreases
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(model, optimizer, epoch, name)

        logging.info(f'Epoch: {epoch:02} | Time: {epoch_time:.2f}m')
        logging.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        logging.info(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}%')

    return history

if __name__ == '__main__':

    # load config
    args = parse_args()
    params = parse_params(args.config)

    # setup hyperparams
    train_df = params.train_df
    valid_df = params.valid_df
    pssm_dir = params.pssm_dir
    tert_dir = params.tert_dir
    max_len = int(params.max_len)
    batch_size = int(params.batch_size)
    input_shape = tuple(int(x) for x in params.input_shape.split())
    n_dist_bins = int(params.n_dist_bins)
    n_blocks = int(params.n_blocks)
    n_epochs = int(params.n_epochs)
    lr = float(params.lr)
    name = params.name

    init_logger(name)
    logging.info(pformat(params))

    # to gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'device: {device}')

    # setup data iterators and model
    train_dataset = ProteinNetDataset(train_df, pssm_dir, tert_dir, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = ProteinNetDataset(valid_df, pssm_dir, tert_dir, max_len)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = ResNet(input_shape, n_dist_bins, n_blocks).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # run training
    run_n_epochs(n_epochs, model, train_loader, valid_loader, optimizer, criterion, name)

