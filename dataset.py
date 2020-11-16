import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import pairwise_distances

class ProteinNet(Dataset):

    def __init__(self, df, pssm_dir, tert_dir, max_len, bins=[4, 6, 8, 10, 12, 14, 16, 18, 20]):

        """dataset for ProteinNet records

        Args:
            df (str): file path for dataframe with sample names and sequences
            pssm_dir (str): folder path for pssm data
            tert_dir (str): folder path for tertiary data
            max_len (int): sequence length to pad up to
            bins ([ints]): bins to group pairwise distances into
        """

        self.bins = bins
        self.max_len = max_len
        self.pssm_dir = pssm_dir
        self.tert_dir = tert_dir

        self.df = pd.read_csv(df)

    def seq_to_ohe(self, seq):

        """convert string sequence into one hot format NxNx20, pad up to max_len

        Args:
            seq (str): amino acid sequence to convert

        Returns:
            ohe_seq (tensor): sequence in one hot format, zeroed on non-matching indices
        """

        key = 'ACDEFGHIKLMNPQRSTVWY'

        pass

    def cat_pssm(self, pssm_ic):

        """construct 2-site pssm feature by concatenating pssm from residue i and residue j,
        1-site information content by matrix mult, add padding

        Args:
            pssm_ic (str): file path for pssm npz

        Returns:
            evo_arr (tensor): pssm and ic features with shape NxNx41
        """

        # 20xN in amino acid alphabetical order, 1xN information content last row

        # transpose to (N, 20)
        pssm = np.load(pssm_ic).T

        # each element cat pssm[i] + pssm[]

        pass

    def tert_to_bins(self, tert):

        """pairwise distances between ca coordinates from tert records, padded

        Args:
            tert (str): file path to tert npz

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

        # pad
        to_pad = self.max_len - len(dist_arr)
        dist_arr = np.pad(dist_arr, (0, to_pad), 'constant')

        # bin distances
        dist_arr = np.digitize(dist_arr, bins=self.bins)

        return dist_arr

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        sample = self.df.iloc[idx]
        sample_name = sample['name']
        sample_seq = sample['seq']
        seq_len = sample['seq_len']

        pssm_path = f'{self.pssm_dir}/{sample_name}.npz'
        tert_path = f'{self.tert_dir}/{sample_name}.npz'


        return sample
