import time

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as I

from data import create_data_loader, load_embedding, create_data_loader_for_small_model, create_pair_loader_for_small_model
from paddlenlp.datasets import GlueCoLA, GlueSST2, GlueMRPC, GlueSTSB, GlueQQP, GlueMNLI, GlueQNLI, GlueRTE, ChnSentiCorp
from paddle.metric import Metric, Accuracy, Precision, Recall

from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman

TASK_CLASSES = {
    "sst-2": (GlueSST2, Accuracy),
    "qqp": (GlueQQP, AccuracyAndF1),
    "mnli": (GlueMNLI, Accuracy),
    "senta": (ChnSentiCorp, Accuracy),
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
        self.embedder.weight.set_value(
            embed_weight) if embed_weight is not None else None

        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers, direction, dropout=dropout_prob)
        self.fc = nn.Linear(
            hidden_size * 2,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))

        self.fc_1 = nn.Linear(
            hidden_size * 8,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))

        self.output_layer = nn.Linear(
            hidden_size,
            output_dim,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x_1, seq_len_1, x_2=None, seq_len_2=None):
        x_embed_1 = self.embedder(x_1)
        lstm_out_1, (hidden_1, _) = self.lstm(
            x_embed_1, sequence_length=seq_len_1)
        out_1 = paddle.concat((hidden_1[-2, :, :], hidden_1[-1, :, :]), axis=1)
        # out = paddle.sum(lstm_out, axis=1)
        if x_2 is not None:
            x_embed_2 = self.embedder(x_2)
            lstm_out_2, (hidden_2, _) = self.lstm(
                x_embed_2, sequence_length=seq_len_2)
            out_2 = paddle.concat(
                (hidden_2[-2, :, :], hidden_2[-1, :, :]), axis=1)
            out = paddle.concat(
                x=[out_1, out_2, out_1 + out_2, paddle.abs(out_1 - out_2)],
                axis=1)
            out = paddle.tanh(self.fc_1(out))
        else:
            out = paddle.tanh(self.fc(out_1))

        logits = self.output_layer(out)
        logits = self.dropout(logits)
        return logits


def evaluate(task_name, model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        if task_name == 'qqp' or task_name == 'mnli':
            input_ids_1, seq_len_1, input_ids_2, seq_len_2, labels = batch
            logits = model(input_ids_1, seq_len_1, input_ids_2, seq_len_2)
        else:
            input_ids, seq_len, labels = batch
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


def do_train(
        task_name='sst-2',
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
        save_steps=20,
        opt='adadelta',
        vocab_path='/root/.paddlenlp/models/bert-base-uncased/bert-base-uncased-vocab.txt',
        use_pretrained_w2v=True):
    metric_class = TASK_CLASSES[task_name][1]
    metric = metric_class()
    if task_name == 'mnli':
        train_data_loader, dev_data_loader_matched, dev_data_loader_mismatched = create_data_loader(
            task_name, batch_size, max_seq_length)
    elif task_name == 'qqp':
        train_data_loader, dev_data_loader = create_pair_loader_for_small_model(
            task_name=task_name,
            batch_size=batch_size,
            language='en',
            vocab_path=vocab_path,
            use_gpu=True)
    elif task_name == 'senta':  # lan: cn
        train_data_loader, dev_data_loader = create_data_loader_for_small_model(
            task_name=task_name,
            batch_size=batch_size,
            language='cn',
            vocab_path=vocab_path,
            use_gpu=True)
    else:
        train_data_loader, dev_data_loader = create_data_loader_for_small_model(
            task_name=task_name,
            batch_size=batch_size,
            language='en',
            vocab_path=vocab_path,
            use_gpu=True)

    emb_tensor = load_embedding(vocab_path) if use_pretrained_w2v else None

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
    if opt == 'adadelta':
        optimizer = paddle.optimizer.Adadelta(
            learning_rate=lr, rho=0.95, parameters=model.parameters())
    else:
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=model.parameters())

    global_step = 0
    tic_train = time.time()
    for epoch in range(num_epoch):
        for i, batch in enumerate(train_data_loader):
            if task_name == 'qqp' or task_name == 'mnli':
                input_ids_1, seq_len_1, input_ids_2, seq_len_2, labels = batch
                logits = model(input_ids_1, seq_len_1, input_ids_2, seq_len_2)
            else:
                input_ids, seq_len, labels = batch
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
                        evaluate(task_name, model, loss_fct, metric,
                                 dev_data_loader_matched)
                        evaluate(task_name, model, loss_fct, metric,
                                 dev_data_loader_mismatched)
                        print("eval done total : %s s" %
                              (time.time() - tic_eval))

                    else:
                        evaluate(task_name, model, loss_fct, metric,
                                 dev_data_loader)
                        print("eval done total : %s s" %
                              (time.time() - tic_eval))
                tic_train = time.time()
            global_step += 1


if __name__ == '__main__':
    vocab_size_dict = {
        "senta": 1256608,
        "bert-base-chinese": 21128,
        "sst-2": 30522,
        "qqp": 30522
    }
    # task_name = 'senta'
    # task_name = 'sst-2'
    paddle.seed(123)
    task_name = 'qqp'
    if task_name == 'senta':
        do_train(
            task_name=task_name,
            vocab_size=vocab_size_dict[task_name],
            batch_size=64,
            lr=2e-4,
            opt='adam',
            num_epoch=10,
            dropout_prob=0.2,
            vocab_path='senta_word_dict.txt',
            use_pretrained_w2v=True)
    elif task_name == 'sst-2':
        do_train(
            task_name=task_name,
            vocab_size=vocab_size_dict[task_name],
            batch_size=64,
            lr=1.0,
            num_epoch=10,
            dropout_prob=0.3,
            use_pretrained_w2v=True)
    else:  # qqp
        do_train(
            task_name=task_name,
            vocab_size=vocab_size_dict[task_name],
            batch_size=256,
            lr=1.0,
            num_epoch=20,
            dropout_prob=0.4,
            use_pretrained_w2v=True)
