import time

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as I

from data import create_data_loader, load_embedding
from paddlenlp.datasets import GlueCoLA, GlueSST2, GlueMRPC, GlueSTSB, GlueQQP, GlueMNLI, GlueQNLI, GlueRTE
from paddle.metric import Metric, Accuracy, Precision, Recall

from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman

TASK_CLASSES = {
    "cola": (GlueCoLA, Mcc),
    "sst-2": (GlueSST2, Accuracy),
    "mrpc": (GlueMRPC, AccuracyAndF1),
    "sts-b": (GlueSTSB, PearsonAndSpearman),
    "qqp": (GlueQQP, AccuracyAndF1),
    "mnli": (GlueMNLI, Accuracy),
    "qnli": (GlueQNLI, Accuracy),
    "rte": (GlueRTE, Accuracy),
}


class BiLSTM(nn.Layer):
    def __init__(self,
                 embed_dim,
                 hidden_size,
                 vocab_size,
                 output_dim,
                 padding_idx=0,
                 num_layers=1,
                 direction='bidirectional',
                 dropout_prob=0.0,
                 init_scale=0.1,
                 embed_weight=None):
        super(BiLSTM, self).__init__()
        self.embedder = nn.Embedding(vocab_size, embed_dim, padding_idx)
        self.embedder.weight.set_value(embed_weight)

        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers, direction, dropout=dropout_prob)
        self.fc = nn.Linear(
            hidden_size * 2,
            output_dim,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, seq_len):
        x_embed = self.embedder(x)
        lstm_out, _ = self.lstm(x_embed, sequence_length=seq_len)
        out = paddle.sum(lstm_out, axis=1)
        logits = self.fc(out)
        logits = self.dropout(logits)
        return logits


def evaluate(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, _, seq_len, labels = batch
        logits = model(input_ids, seq_len)
        loss = loss_fct(logits, labels)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    if isinstance(metric, AccuracyAndF1):
        print(
            "eval loss: %f, acc: %s, precision: %s, recall: %s, f1: %s, acc and f1: %s, "
            % (
                loss.numpy(),
                res[0],
                res[1],
                res[2],
                res[3],
                res[4], ),
            end='')
    elif isinstance(metric, Mcc):
        print("eval loss: %f, mcc: %s, " % (loss.numpy(), res[0]), end='')
    elif isinstance(metric, PearsonAndSpearman):
        print(
            "eval loss: %f, pearson: %s, spearman: %s, pearson and spearman: %s, "
            % (loss.numpy(), res[0], res[1], res[2]),
            end='')
    else:
        print("eval loss: %f, acc: %s, " % (loss.numpy(), res), end='')
    model.train()


def do_train(task_name='sst-2',
             num_epoch=30,
             batch_size=128,
             lr=1.0,
             max_seq_length=128,
             emb_dim=300,
             hidden_size=300,
             output_dim=2,
             vocab_size=30522,
             padding_idx=0,
             dropout_prob=0.3,
             save_steps=20):
    metric_class = TASK_CLASSES[task_name][1]
    metric = metric_class()
    if task_name == 'mnli':
        train_data_loader, dev_data_loader_matched, dev_data_loader_mismatched = create_data_loader(
            task_name, batch_size, max_seq_length)
    else:
        train_data_loader, dev_data_loader = create_data_loader(
            task_name, batch_size, max_seq_length)

    emb_tensor = load_embedding()
    model = BiLSTM(
        emb_dim,
        hidden_size,
        vocab_size,
        output_dim,
        padding_idx,
        dropout_prob=dropout_prob,
        embed_weight=emb_tensor)

    loss_fct = nn.CrossEntropyLoss()
    # gloabl_norm_clip = paddle.nn.ClipGradByNorm(5.0)

    # optimizer = paddle.optimizer.Adam(
    #     learning_rate=lr,
    #     parameters=model.parameters())
    optimizer = paddle.optimizer.Adadelta(
        learning_rate=lr, rho=0.95, parameters=model.parameters())

    global_step = 0
    tic_train = time.time()
    for epoch in range(num_epoch):
        for i, batch in enumerate(train_data_loader):
            input_ids, _, seq_len, labels = batch
            logits = model(input_ids, seq_len)

            loss = loss_fct(logits, labels)
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

            if i % save_steps == 0:
                with paddle.no_grad():
                    print(
                        "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.4f step/s"
                        % (global_step, epoch, i, loss,
                           save_steps / (time.time() - tic_train)))
                    tic_eval = time.time()
                    if task_name == 'mnli':
                        evaluate(model, loss_fct, metric,
                                 dev_data_loader_matched)
                        evaluate(model, loss_fct, metric,
                                 dev_data_loader_mismatched)
                        print("eval done total : %s s" %
                              (time.time() - tic_eval))

                    else:
                        evaluate(model, loss_fct, metric, dev_data_loader)
                        print("eval done total : %s s" %
                              (time.time() - tic_eval))
                tic_train = time.time()
            global_step += 1


if __name__ == '__main__':
    do_train()
