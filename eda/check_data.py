import pandas as pd
import numpy as np
import subprocess
import shlex
from tqdm.auto import tqdm

def get_sample_headers(fn):

    '''get sample name and amino acid sequence data'''

    cmd = f"grep '\[ID' -A 3 {fn}"
    headers = subprocess.check_output(shlex.split(cmd)).decode('utf-8').split()

    res = []
    for i, line in enumerate(headers):
        if '[ID]' in line:
            name = headers[i+1]
            seq = headers[i+3]
            seq_len = len(seq)

            res.append((name, seq, seq_len))

    df = pd.DataFrame(res, columns=['name', 'seq', 'len'])
    df['fn'] = fn

    return df


def main():

    proteinnet = ['testing', 'training_30', 'training_50', 'training_70', 'training_90', 'training_95', 'training_100', 'validation']

    df_list = []
    for fn in tqdm(proteinnet):
        df = get_sample_headers(fn)
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)
    df.to_csv('sample_header.csv', index=False)

    return df

if __name__ == '__main__':

    main()
