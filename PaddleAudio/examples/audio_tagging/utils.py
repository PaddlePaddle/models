import logging
import paddle
import yaml
import numpy as np
import os
import json
from sklearn.metrics import average_precision_score
import paddle.nn.functional as F

__all__ = [
    'get_logger', 'save_checkpoint', 'load_checkpoint', 'get_labels527', 'random_choice', 'get_label_name_mapping'
]


def random_choice(a):
    i = np.random.randint(0, high=len(a), size=(1, ))
    return a[int(i)]


def get_labels527():
    with open(c['audioset_label']) as F:
        labels527 = F.read().split('\n')
    return labels527


def get_ytid_clsidx_mapping():
    """
    Compute the mapping between youtube id and class index. 
    The class index range from 0 to 526, correspoding to the labels stored in audioset_labels527.txt file
    """
    labels527 = get_labels527()
    label2clsidx = {l: i for i, l in enumerate(labels527)}
    lines = open('./assets/unbalanced_train_segments.csv').read().split('\n')
    lines += open('./assets/balanced_train_segments.csv').read().split('\n')
    lines += open('./assets/eval_segments.csv').read().split('\n')
    lines = [l for l in lines if len(l) > 0 and l[0] != '#']
    ytid2clsidx = {}
    for l in lines:
        ytid = l.split(',')[0]
        labels = l.split(',')[3:]
        cls_idx = []
        for label in labels:
            label = label.replace('"', '').strip()
            cls_idx.append(label2clsidx[label])
        ytid2clsidx.update({ytid: cls_idx})
    clsidx2ytid = {i: [] for i in range(527)}
    for k in ytid2clsidx.keys():
        for v in ytid2clsidx[k]:
            clsidx2ytid[v] += [k]
    return ytid2clsidx, clsidx2ytid


def get_metrics(label, pred):
    a = label.numpy()
    b = (pred.numpy() > 0.5).astype('int32')
    eps = 1e-8
    tp = np.sum(b[a == 1])
    fp = np.sum(b[a == 0])
    precision = tp / (fp + tp + eps)
    fn = np.sum(b[a == 1] == 0)
    recall = tp / (tp + fn)
    return precision, recall


def get_logger(name, log_path, level='INFO', fmt='%(asctime)s-%(levelname)s: %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.FileHandler(log_path, mode='a')
    formatter = logging.Formatter(fmt=fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def save_checkpoint(step, model, optimizer, prefix):
    logger.info('checkpointing at step', step)
    last_check = c['model_path'] + '/{}_checkpoint{}.pdparams'.format(prefix, step - c['checkpoint_step'])
    if os.path.exists(last_check):
        os.system('rm ' + last_check)
    last_check = c['model_path'] + '/{}_checkpoint{}.pdopt'.format(prefix, step - c['checkpoint_step'])
    if os.path.exists(last_check):
        os.system('rm ' + last_check)

    paddle.save(model.state_dict(), c['model_path'] + '/{}_checkpoint{}.pdparams'.format(prefix, step))
    paddle.save(optimizer.state_dict(), c['model_path'] + '/{}_checkpoint{}.pdopt'.format(prefix, step))


def load_checkpoint(epoch, prefix):
    file = c['model_path'] + '/{}_checkpoint_model{}.tar'.format(prefix, epoch)
    logger.info('loading checkpoing ' + file)
    model_dict = paddle.load(c['model_path'] + '/{}_checkpoint{}.pdparams'.format(prefix, epoch))
    optim_dict = paddle.load(c['model_path'] + '/{}_checkpoint{}.pdopt'.format(prefix, epoch))
    return model_dict, optim_dict


def evaluate(epoch, val_loader, model, loss_fn, log_writer):
    model.eval()
    avg_loss = 0.0
    avg_preci = 0.0
    avg_recall = 0.0
    all_labels = []
    all_preds = []
    for batch_id, data in enumerate(val_loader()):
        xd, yd = data
        xd = xd.unsqueeze((1))
        label = yd
        logits = model(xd)
        loss_val = loss_fn(logits, label)
        pred = F.softmax(logits)
        all_labels += [label.numpy()]
        all_preds += [pred.numpy()]

        preci, recall = get_metrics(label, pred)
        avg_loss = (avg_loss * batch_id + loss_val.numpy()[0]) / (1 + batch_id)
        avg_preci = (avg_preci * batch_id + preci) / (1 + batch_id)
        avg_recall = (avg_recall * batch_id + recall) / (1 + batch_id)

        msg = f'eval epoch:{epoch}, batch:{batch_id}'
        msg += f'|{len(val_loader)}'
        msg += f',loss:{avg_loss:.3}'
        msg += f',recall:{avg_recall:.3}'
        msg += f',preci:{avg_preci:.3}'
        avg_preci = (avg_preci * batch_id + preci) / (1 + batch_id)
        avg_recall = (avg_recall * batch_id + recall) / (1 + batch_id)
        if batch_id % 20 == 0:
            logger.info(msg)
            log_writer.add_scalar(tag="eval loss", step=batch_id, value=avg_loss)
            log_writer.add_scalar(tag="eval preci", step=batch_id, value=avg_preci)
            log_writer.add_scalar(tag="eval recall", step=batch_id, value=avg_recall)

    all_preds = np.concatenate(all_preds, 0)
    all_labels = np.concatenate(all_labels, 0)
    mAP_scores = average_precision_score(all_labels, all_preds, average=None)

    return avg_loss, avg_preci, avg_recall, mAP_scores


def get_label_name_mapping():
    with open('./assets/ontology.json') as F:
        ontology = json.load(F)
    label2name = {o['id']: o['name'] for o in ontology}
    name2label = {o['name']: o['id'] for o in ontology}
    return label2name, name2label


def download_assets():
    os.makedirs('./assets/', exist_ok=True)
    urls = [
        'https://raw.githubusercontent.com/audioset/ontology/master/ontology.json',
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv',
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv',
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv'
    ]
    for url in urls:
        fname = './assets/' + url.split('/')[-1]
        if os.path.exists(fname):
            continue

        cmd = 'wget ' + url + ' -O ' + fname
        print(cmd)
        os.system(cmd)


with open('./config.yaml') as f:
    c = yaml.safe_load(f)
logger = get_logger(__name__, os.path.join(c['log_path'], 'log.txt'))
