from pathlib import Path
import numpy as np
import pandas as pd

def save_training(fn, n_save, max_len):

    '''subset record text from training split'''

    # save pssm and coordinates
    path_name = Path(f'{fn}_data')
    if not path_name.exists():
        path_name.mkdir()

    res = []
    n_saved = 0

    with open(fn, 'r') as content:
        lines = content.readlines()

    for i, line in enumerate(lines):
        if '[ID]' in line:
            start = i
        if '[PRIMARY]' in line:
            seq = lines[i+1]
            if len(seq) <= max_len:

                # save whole record
                res.append(''.join(lines[start:start + 33]))
                n_saved += 1

                if n_saved == n_save:
                    break

    with open(Path(path_name, 'filtered.txt'), 'w') as fo:
        fo.writelines(res)

    return res


def save_pssm_tert(fn):

    '''preprocess and save pssm/tertiary coordinates to npy files'''

    content = Path(fn).read_text().splitlines()

    res = []
    for i, line in enumerate(content):

        # save df with id name and sequence
        if '[ID]' in line:
            name = content[i+1].strip()
            seq = content[i+3].strip()
            seq_len = len(seq)

            # include mask
            mask = content[i+31].strip()

            res.append([name, seq, seq_len, mask])

        # save pssm and tert as npz
        if '[EVOLUTIONARY]' in line:

            pssm_str = ' '.join(content[i+1: i+22])
            pssm = np.fromstring(pssm_str, sep=' ')

            pssm_path = f'pssm/{name}_pssm'
            np.save(pssm_path, pssm)

        if '[TERTIARY]' in line:

            tert_str = ' '.join(content[i+1: i+4])
            tert = np.fromstring(tert_str, sep= ' ')

            tert_path = f'tert/{name}_tert'
            np.save(tert_path, tert)


    df = pd.DataFrame(res, columns=['name', 'seq', 'seq_len', 'mask'])

    return df


if __name__ == '__main__':

    # max len 250
    # save 5000 from training_30
    save_training('training_30', 5000, 250)
    df = save_pssm_tert('training_30/filtered.txt')
    df.to_csv('training_30_dataset.csv', index=False)

    # save validation
    save_training('validation', 5000, 250)
    df = save_pssm_tert('validation_data/filtered.txt')
    df.to_csv('validation_dataset.csv', index=False)

    # save testing
    save_training('testing', 5000, 250)
    df = save_pssm_tert('testing_data/filtered.txt')
    df.to_csv('testing_dataset.csv', index=False)
