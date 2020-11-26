from utils import load_checkpoint, contact_precision
from collections import defaultdict 
import itertools

contact_types = ['short', 'med', 'long']
tops = ['l', 'l/2', 'l/5']
metrics = itertools.product(contact_types, tops)

def run_test(model, loader, criterion):

    model.eval()

    epoch_loss = 0
    epoch_acc = 0

    # save contact precision metrics
    contacts = defaultdict(list)

    with torch.no_grad():

        for batch_idx, (feats, dist_arr, seq_len) in enumerate(loader):

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

            for metric in metrics:
                metric_name = f'{metric[0]}_{metric[1]}'
                res = contact_precision(out, dist_arr, seq_len, metric[0], metric[1])

                contacts[metric_name] = res.item()

    return epoch_loss / len(loader), epoch_acc / len(loader), contact

