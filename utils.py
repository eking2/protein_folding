import yaml
import logging
import torch
import torch.nn.functional as F
from pathlib import Path

class dotdict(dict):

    '''dot.notation access to dict attributes'''

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_params(yaml_path):

    '''Read hyperparameters from config yaml'''

    content = Path(yaml_path).read_text()
    params = yaml.safe_load(content)

    return dotdict(params)


def init_logger(log_file):

    '''setup logger to print to screen and save to file'''

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handlers = [logging.StreamHandler(),
                logging.FileHandler(f'logs/{log_file}.log', 'a')]

    # do not print millisec
    fmt = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s',
                            "%Y-%m-%d %H:%M:%S")

    for h in handlers:
        h.setFormatter(fmt)
        logger.addHandler(h)

    return logger


def save_checkpoint(model, optimizer, epoch, file_name, delete=True):

    '''save model state dict and optimizer state dict'''

    # remove last checkpoint file
    if delete:

        # first checkpoint will not have file
        try:
            old_check = list(Path('checkpoints').glob('*.pt'))[0]
            old_check.unlink()
        except:
            pass

    torch.save({'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict()},
        f'checkpoints/{file_name}_{epoch}.pt')


def load_checkpoint(checkpoint, model, optimizer=None):

    '''load model state dict to continue training or evaluate'''

    check = Path(checkpoint)
    if not check.exists():
        raise Exception(f'File does not exist {checkpoint}')

    check = torch.load(check, map_location='cpu')
    model.load_state_dict(check['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def get_mask(seq_len, labels, contact, max_len, device):

    '''select region to calculate contact precision over'''

    # make grid
    seq_range = torch.arange(seq_len.item())
    xx, yy = torch.meshgrid(seq_range, seq_range)
    
    # distance in sequence
    # 6 <= x <= 11
    if contact == 'short':
        mask = (torch.abs(xx - yy) >= 6) & (torch.abs(xx - yy) <= 11)
    
    # 12 <= x <= 23
    elif contact == 'med':
        mask = (torch.abs(xx - yy) >= 12) & (torch.abs(xx - yy) <= 23)
        
    # >= 24
    else:
        mask = torch.abs(xx - yy) >= 24

    mask = mask.to(device)
        
    # set contact bin probs
    # 0-8 A
    condition = (labels <= 3) & (labels >= 0)
    contact_map = torch.where(condition, torch.tensor(1.).to(device), torch.tensor(0.).to(device)).view(1, max_len, max_len)
    
    # selection mask
    # ignore disordered
    not_disordered = torch.where(labels != -1, torch.tensor(1.).to(device), torch.tensor(0.).to(device))
    to_pad = max_len - seq_len.item()
    mask = (F.pad(mask, (0, to_pad, 0, to_pad)) * not_disordered).view(1, max_len, max_len)

    return mask, contact_map


def contact_precision(model_out, labels, seq_len, contact, top, device, max_len=250):

    '''calculate contact precision over selected sequence distance and length range'''

    assert contact in ['short', 'med', 'long'], 'invalid contact'
    assert top in ['l', 'l/2', 'l/5'], 'invalid top'

    mask, contact_map = get_mask(seq_len, labels, contact, max_len, device)
    mask = mask.view(1, max_len, max_len)
    contact_map = contact_map.view(1, max_len, max_len)

    # softmax on model output, logits to preds
    softmaxed = F.softmax(model_out, dim=1)
    
    # add probabilities for bins 0-8A 
    probs = softmaxed[:, :3, :, :].sum(dim=1)
    
    # mask out
    probs = probs * mask
    
    # select number of samples
    if top == 'l/2':
        res_range = seq_len // 2
    elif top == 'l/5':
        res_range = seq_len // 5
    else:
        res_range = seq_len

    res_range = res_range.item()
    
    # get index of most confident predictions
    # argsort and order decreasing
    probs = probs.flatten()
    probs_idx = torch.argsort(probs, descending=True)[:res_range]

    # from probability to binary contact prediction
    pred = torch.tensor([1 if probs[idx] >= 0.5 else 0 for idx in probs_idx]).to(device)
    
    # get true labels at selected indices
    sel_contacts = contact_map.flatten()[probs_idx]

    # compare
    correct = (pred == sel_contacts).sum().float()
    acc = (correct / res_range * 100).item()
    
    return acc, correct.long().item(), res_range
