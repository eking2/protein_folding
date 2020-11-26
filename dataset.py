import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import pairwise_distances

class ProteinNetDataset(Dataset):

    def __init__(self, df, pssm_dir, tert_dir, max_len, bins=[4, 6, 8, 10, 12, 14, 16, 18, 20]):

        """dataset for ProteinNet records

        Args:
            df (str): file path for dataframe with sample names and sequences
            pssm_dir (str): folder path for pssm data
            tert_dir (str): folder path for tertiary data
            max_len (int): sequence length to pad up to
            bins ([ints]): bins to group pairwise distances into, default 10 CASP14
        """

        self.bins = bins
        self.max_len = max_len
        self.pssm_dir = pssm_dir
        self.tert_dir = tert_dir

        self.df = pd.read_csv(df)

    def seq_to_ohe(self, seq):

        """convert string sequence into 2-site one hot format NxNx42, +2 from padding

        Args:
            seq (str): amino acid sequence to convert

        Returns:
            ohe_seq (tensor): sequence in 2-site one hot format, cat seq[i] + seq[j]
        """

        # vocab
        key = 'ACDEFGHIKLMNPQRSTVWY'
        key_chars = len(key) + 1
        self.itos = dict(enumerate(key, 1))  # reserve 0 for padding
        self.stoi = {char: i for i, char in self.itos.items()}

        # convert all sequences to labels
        label_encoded = [self.stoi[char] for char in seq]

        # one hot
        ohe = np.eye(key_chars)[label_encoded]

        # pad
        to_pad = self.max_len - len(ohe)
        ohe = np.concatenate([ohe, np.zeros((to_pad, key_chars))])

        # cat seq[i] + seq[j]
        ohe_seq = np.zeros((self.max_len, self.max_len, key_chars*2 ))

        # (N, 21) to (N, N, 21)
        # same as np.tile(ohe, (N, 1, 1)) or np.broadcast_to(ohe, (N, N, 21))
        ohe_repeat = np.repeat(np.expand_dims(ohe, 0), self.max_len, axis=0)  

        ohe_seq[:, :, :21] = ohe_repeat
        ohe_seq[:, :, 21:] = np.transpose(ohe_repeat, (1, 0, 2))

        return ohe_seq


    def cat_pssm(self, pssm_ic):

        """construct 2-site pssm feature by concatenating pssm from residue i and residue j,
        1-site information content by matrix mult, add padding

        Args:
            pssm_ic (str): file path for pssm npy

        Returns:
            evo_arr (tensor): pssm and ic features with shape NxNx42
        """

        # 20xN in amino acid alphabetical order, 1xN information content last row

        # reshape flat to (N, 21) 
        pssm_ic = np.load(pssm_ic).reshape(21, -1).T

        # pad
        to_pad = self.max_len - len(pssm_ic)
        pssm_ic = np.concatenate([pssm_ic, np.zeros((to_pad, 21))])

        # each element cat pssm_ic[i] + pssm_ic[j]
        evo_arr = np.zeros((self.max_len, self.max_len, 42))

        # (N, 21) to (N, N, 21)
        pssm_repeat = np.repeat(np.expand_dims(pssm_ic, 0), self.max_len, axis=0)

        evo_arr[:, :, :21] = pssm_repeat
        evo_arr[:, :, 21:] = np.transpose(pssm_repeat, (1, 0, 2))

        return evo_arr

    def tert_to_bins(self, tert, mask):

        """pairwise distances between ca coordinates from tert records, padded

        Args:
            tert (str): file path to tert npy
            mask (str): mask sequence indicating (-) disordered or (+) ordered atoms

        Returns:
            dist_arr (tensor): binned pairwise distance matrix with shape NxNx1
        """
        # cols: n, ca, c
        # rows: x, y, z

        # to angstrom
        coords = np.load(tert).reshape(3, -1) / 100

        # keep only ca
        ca = coords.transpose()[1::3]

        # pairwise distances
        dist_arr = pairwise_distances(ca)

        # bin distances
        dist_arr = np.digitize(dist_arr, bins=self.bins)

        # index to mask out with -1
        mask_idx = np.array([i for i, x in enumerate(mask) if x == '-'])

        # skip fully ordered samples
        if sum(mask_idx) > 0:
 
            # mask out
            dist_arr[mask_idx, :] = -1
            dist_arr[:, mask_idx] = -1

        # pad
        to_pad = self.max_len - len(dist_arr)
        dist_arr = np.pad(dist_arr, (0, to_pad), 'constant', constant_values=-1)

        return dist_arr

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        sample = self.df.iloc[idx]
        sample_name = sample['name']
        sample_seq = sample['seq']
        seq_len = sample['seq_len']
        mask = sample['mask']

        pssm_path = f'./{self.pssm_dir}/{sample_name}_pssm.npy'
        tert_path = f'./{self.tert_dir}/{sample_name}_tert.npy'

        # (N, N, 42)
        ohe_seq = self.seq_to_ohe(sample_seq)

        # (N, N, 42)
        evo_arr = self.cat_pssm(pssm_path)

        # (N, N, 1) -> (1, N, N)
        # mask dist_arr with -1 to ignore padding and disordered atoms
        dist_arr = np.expand_dims(self.tert_to_bins(tert_path, mask), 2).reshape(1, self.max_len, self.max_len)

        # cat features and reshape channel first
        # (84, N, N)
        feats = np.concatenate([ohe_seq, evo_arr], axis=2).transpose(2, 0, 1)

        return feats, dist_arr, seq_len


def test():

    df = 'input/small_dataset.csv'
    pssm_dir = 'pssm'
    tert_dir = 'tert'
    max_len = 250

    dataset = ProteinNetDataset(df, pssm_dir, tert_dir, max_len)
    feats, dist_arr = dataset[8]
    
    import matplotlib.pyplot as plt

    plt.imshow(dist_arr, cmap='viridis_r')
    plt.colorbar()
    plt.savefig('test.png', dpi=300)

    print(sample_name)
    #print(feats)
    print(feats.shape)
    #print(dist_arr)
    print(dist_arr.shape)

if __name__ == '__main__':
    test()

