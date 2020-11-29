from utils import load_checkpoint, contact_precision
from collections import defaultdict 
from dataset import ProteinNetDataset
from model import ResNet
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import itertools
import numpy as np
import pandas as pd

def run_test(model, loader, criterion, metrics, device):

    model.eval()

    epoch_loss = 0
    epoch_acc = 0

    contacts = defaultdict(list)

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
            
            contacts['name'].append(name[0])
            contacts['acc'].append(acc.item() * 100)

            # output the true contact and pred contact for viz
            name = name[0].replace('#', '_')
            np.save(f'debug/{name}_pred', pred.cpu().numpy())
            np.save(f'debug/{name}_true', dist_arr.cpu().numpy())

            # contact precision for each sample
            for metric in metrics:
                metric_name = f'{metric[0]}_{metric[1]}'

                res = contact_precision(out, dist_arr, seq_len, metric[0], metric[1], device)
                contacts[metric_name].append(res)

    contact_df = pd.DataFrame.from_dict(contacts, orient='columns').round(3)

    return epoch_loss / len(loader), epoch_acc / len(loader), contact_df



if __name__ == '__main__':

    contact_types = ['short', 'med', 'long']
    tops = ['l', 'l/2', 'l/5']
    metrics = list(itertools.product(contact_types, tops))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ProteinNetDataset('./input/testing_dataset.csv', 'pssm', 'tert', 250)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    input_shape = (1, 84, 250, 250)
    n_dist_bins = 10
    n_blocks = 4

    model = ResNet(input_shape, n_dist_bins, n_blocks)
    load_checkpoint('./checkpoints/train_config_20.pt', model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    loss, acc, df = run_test(model, loader, criterion, metrics, device)
    print(df)
