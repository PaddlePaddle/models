import time

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as I

from data import create_data_loader, evaluate
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


class CrossEntropyCriterion(nn.Layer):
    def __init__(self):
        super(CrossEntropyCriterion, self).__init__()

    def forward(self, predict, label, mask):
        cost = F.softmax_with_cross_entropy(
            logits=predict, label=label, soft_label=False)
        cost = paddle.squeeze(cost, axis=[1])
        masked_cost = cost * mask
        batch_mean_cost = paddle.mean(masked_cost, axis=[0])
        seq_cost = paddle.sum(batch_mean_cost)
        return seq_cost


class BiLSTM(nn.Layer):
    def __init__(self,
                 embed_dim,
                 hidden_size,
                 vocab_size,
                 output_dim,
                 padding_idx=0,
                 num_layers=1,
                 direction='bidirectional',
                 dropout_prob=0.5,
                 init_scale=0.1):
        super(BiLSTM, self).__init__()
        self.embedder = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers, direction, dropout=dropout_prob)
        self.fc = nn.Linear(
            hidden_size * 2,
            output_dim,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, seq_len):
        x_embed = self.dropout(self.embedder(x))
        lstm_out, _ = self.lstm(x_embed, sequence_length=seq_len)
        hidden = paddle.mean(lstm_out, axis=1)
        # logits = self.fc(lstm_out[:, -1, :])
        logits = self.fc(hidden)

        return logits


def do_train(task_name='sst-2',
             num_epoch=20,
             batch_size=128,
             lr=0.5,
             max_seq_length=128,
             emb_dim=256,
             hidden_size=256,
             output_dim=2,
             vocab_size=30522,
             padding_idx=0,
             dropout_prob=0.5,
             save_steps=100):
    metric_class = TASK_CLASSES[task_name][1]
    metric = metric_class()
    train_data_loader, dev_data_loader = create_data_loader(
        task_name, batch_size, max_seq_length)
    # from paddlenlp.models.senta import LSTMModel
    # model = LSTMModel(vocab_size=vocab_size, num_classes=2, emb_dim=emb_dim, padding_idx=padding_idx, lstm_hidden_size=hidden_size, direction='bidirectional', dropout_rate=dropout_prob, fc_hidden_size=hidden_size / 2)
    # model = LSTMModel(vocab_size=vocab_size, num_classes=2)
    model = BiLSTM(emb_dim, hidden_size, vocab_size, output_dim, padding_idx)
    # loss_fct = nn.CrossEntropyLoss()#NLLLoss()
    loss_fct = CrossEntropyCriterion()
    adam = paddle.optimizer.SGD(learning_rate=lr, parameters=model.parameters())

    loss_list = []
    global_step = 0
    tic_train = time.time()
    for epoch in range(num_epoch):
        metric.reset()
        for i, batch in enumerate(train_data_loader):
            input_ids, _, seq_len, labels = batch
            # input_ids = input_ids[:, 1:]
            logits = model(input_ids, seq_len)

            loss = loss_fct(logits, labels, seq_len)
            adam.clear_grad()
            loss.backward()
            adam.step()
            correct = metric.compute(F.log_softmax(logits), labels)
            metric.update(correct)

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
        res = metric.accumulate()
        print(res)


if __name__ == '__main__':
    do_train()
