import time

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.metric import Metric, Accuracy, Precision, Recall

from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.transformers.tokenizer_utils import whitespace_tokenize
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
from paddlenlp.datasets import GlueCoLA, GlueSST2, GlueMRPC, GlueSTSB, GlueQQP, GlueMNLI, GlueQNLI, GlueRTE

from small import BiLSTM
from data import create_data_loader, evaluate

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


class Teacher(object):
    def __init__(self,
                 model_name='bert-base-uncased',
                 param_path='model/SST-2/best_model_350/model_state.pdparams',
                 max_seq_len=128):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.model.set_state_dict(paddle.load(param_path))
        self.model.eval()


def evaluate(model,
             ce_loss,
             mse_loss,
             metric,
             data_loader,
             teacher_eval_logits_list,
             alpha=0.5):
    model.eval()
    metric.reset()
    for i, batch in enumerate(data_loader):
        teacher_logits = teacher_eval_logits_list[i]
        input_ids, _, seq_len, labels = batch
        logits = model(input_ids, seq_len)
        loss = alpha * ce_loss(logits, labels) + (1 - alpha) * mse_loss(
            logits, teacher_logits)
        # F.softmax(logits), F.softmax(teacher_logits))

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
             num_epoch=60,
             batch_size=128,
             lr=0.005,
             max_seq_length=128,
             emb_dim=256,
             hidden_size=256,
             output_dim=2,
             vocab_size=30522,
             padding_idx=0,
             dropout_prob=0.5,
             save_steps=50,
             alpha=0):

    train_data_loader, dev_data_loader = create_data_loader(
        task_name, batch_size, max_seq_length, shuffle=False)

    model = BiLSTM(
        emb_dim,
        hidden_size,
        vocab_size,
        output_dim,
        padding_idx,
        dropout_prob=dropout_prob)
    # adam = paddle.optimizer.SGD(learning_rate=lr, parameters=model.parameters())#, grad_clip=gloabl_norm_clip)

    gloabl_norm_clip = paddle.nn.ClipGradByNorm(5.0)
    adam = paddle.optimizer.Adam(
        learning_rate=lr, parameters=model.parameters())
    # grad_clip=gloabl_norm_clip)

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    metric_class = TASK_CLASSES[task_name][1]
    metric = metric_class()

    teacher = Teacher()
    teacher_train_logits_list = []
    teacher_eval_logits_list = []
    with paddle.no_grad():
        for i, batch in enumerate(train_data_loader):
            input_ids, segment_ids, _, labels = batch
            teacher_logits = teacher.model(input_ids, segment_ids)
            teacher_train_logits_list.append(teacher_logits)
        for i, batch in enumerate(dev_data_loader):
            input_ids, segment_ids, _, labels = batch
            teacher_logits = teacher.model(input_ids, segment_ids)
            teacher_eval_logits_list.append(teacher_logits)

    print("Teacher model's eval logits have been calculated. Start to train...")
    global_step = 0
    tic_train = time.time()
    for epoch in range(num_epoch):
        model.train()
        for i, batch in enumerate(train_data_loader):
            input_ids, segment_ids, seq_len, labels = batch
            # with paddle.no_grad():
            #     teacher.model.eval()
            #     teacher_logits = teacher.model(input_ids, segment_ids)
            logits = model(input_ids, seq_len)
            model.train()
            teacher_logits = teacher_train_logits_list[i]
            loss = alpha * ce_loss(logits, labels) + (1 - alpha) * mse_loss(
                logits, teacher_logits)

            loss.backward()
            adam.step()
            adam.clear_grad()

            global_step += 1
            if i % save_steps == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.4f step/s"
                    % (global_step, epoch, i, loss,
                       save_steps / (time.time() - tic_train)))
                tic_eval = time.time()
                if task_name == 'mnli':
                    evaluate(model, ce_loss, mse_loss, metric,
                             dev_data_loader_matched, teacher_eval_logits_list,
                             alpha)
                    evaluate(model, ce_loss, mse_loss, metric,
                             dev_data_loader_mismatched, teacher_logits_list,
                             teacher_eval_logits_list, alpha)
                    print("eval done total : %s s" % (time.time() - tic_eval))

                else:
                    evaluate(model, ce_loss, mse_loss, metric, dev_data_loader,
                             teacher_eval_logits_list, alpha)
                    print("eval done total : %s s" % (time.time() - tic_eval))
                tic_train = time.time()


if __name__ == '__main__':
    do_train()
