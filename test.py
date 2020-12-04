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
import argparse
from utils import parse_params

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='test config path')

    return parser.parse_args()

def run_test(model, loader, criterion, metrics, device):

    '''test model and evaluate performance through contact precision'''

    model.eval()

    epoch_loss = 0
    epoch_acc = 0

    # per sample statistics
    contacts = defaultdict(list)

    # stats for all pairwise comparisons
    agg_stats = defaultdict(list)

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

            agg_stats['name'].append(name[0])

            # output the true contact and pred contact for viz
            name = name[0].replace('#', '_')
            np.save(f'debug/{name}_pred', pred.cpu().numpy())
            np.save(f'debug/{name}_true', dist_arr.cpu().numpy())

            # contact precision for each sample
            for metric in metrics:
                metric_name = f'{metric[0]}_{metric[1]}'

                acc, correct, res_range = contact_precision(out, dist_arr, seq_len, metric[0], metric[1], device)
                contacts[metric_name].append(acc)
                agg_stats[f'{metric_name}_cor'].append(correct)
                agg_stats[f'{metric_name}_len'].append(res_range)

    contact_df = pd.DataFrame.from_dict(contacts, orient='columns').round(3)
    agg_df = pd.DataFrame.from_dict(agg_stats, orient='columns')

    return epoch_loss / len(loader), epoch_acc / len(loader), contact_df, agg_df


def get_precision_stats(agg_df, contact_types):

    '''calculate precision stats over all pairwise predictions (not by each sample)'''

    df = agg_df.copy()

    # drop name
    df = df.drop('name', axis=1)

    df = df.sum(axis=0).to_frame().reset_index()
    df.columns = ['metric', 'value']
    df['metric_type'] = df['metric'].str.rsplit('_', 1, expand=True)[1]
    df['metric'] = df['metric'].str.rsplit('_', 1, expand=True)[0]

    def calc_precisions(group):

        correct = group.query("metric_type == 'cor'")['value'].values[0]
        length = group.query("metric_type == 'len'")['value'].values[0]

        return correct / length

    df_agg = df.groupby('metric').apply(calc_precisions)
    df_agg.to_csv('contact_precisions.csv')


if __name__ == '__main__':

    args = parse_args()
    params = parse_params(args.config)

    test_df = params.test_df
    pssm_dir = params.pssm_dir
    tert_dir = params.tert_dir
    max_len = int(params.max_len)
    batch_size = int(params.batch_size)
    n_dist_bins = int(params.n_dist_bins)
    n_blocks = int(params.n_blocks)
    checkpoint = params.checkpoint
    input_shape = tuple(int(x) for x in params.input_shape.split())

    name = checkpoint.split('/')[-1][:-4]

    contact_types = ['short', 'med', 'long']
    tops = ['l', 'l/2', 'l/5']
    metrics = list(itertools.product(contact_types, tops))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ProteinNetDataset(test_df, pssm_dir, tert_dir, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ResNet(input_shape, n_dist_bins, n_blocks)
    load_checkpoint(checkpoint, model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    loss, acc, contact_df, agg_df = run_test(model, loader, criterion, metrics, device)
    get_precision_stats(agg_df, contact_types)

    contact_df.to_csv(f'output/{name}_contact_df.csv', index=False)
    agg_df.to_csv(f'output/{name}_agg_df.csv', index=False)

