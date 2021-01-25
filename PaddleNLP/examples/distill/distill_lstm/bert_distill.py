import time

import paddle
import paddle.nn as nn
from paddle.metric import Metric, Accuracy, Precision, Recall

from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.transformers.tokenizer_utils import whitespace_tokenize
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
from paddlenlp.datasets import GlueCoLA, GlueSST2, GlueMRPC, GlueSTSB, GlueQQP, GlueMNLI, GlueQNLI, GlueRTE, ChnSentiCorp

from small import BiLSTM
from data import create_distill_loader, load_embedding

TASK_CLASSES = {
    "sst-2": (GlueSST2, Accuracy),
    "qqp": (GlueQQP, AccuracyAndF1),
    "mnli": (GlueMNLI, Accuracy),
    "senta": (ChnSentiCorp, Accuracy),
}


class Teacher(object):
    def __init__(self,
                 model_name='bert-base-uncased',
                 param_path='model/SST-2/best_model_610/model_state.pdparams',
                 max_seq_len=128):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.model.set_state_dict(paddle.load(param_path))
        self.model.eval()


def evaluate(task_name,
             model,
             ce_loss,
             mse_loss,
             metric,
             data_loader,
             teacher_eval_logits_list,
             alpha=0.0):
    model.eval()
    metric.reset()
    for i, batch in enumerate(data_loader):
        teacher_logits = teacher_eval_logits_list[i]
        if task_name == 'qqp' or task_name == 'mnli':
            _, _, small_input_ids_1, seq_len_1, small_input_ids_2, seq_len_2, labels = batch
            logits = model(small_input_ids_1, seq_len_1, small_input_ids_2,
                           seq_len_2)
        else:
            _, _, small_input_ids, seq_len, labels = batch
            logits = model(small_input_ids, seq_len)

        loss = alpha * ce_loss(logits, labels) + (1 - alpha) * mse_loss(
            logits, teacher_logits)

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
        language='en',
        num_epoch=6,
        batch_size=128,
        opt='adadelta',
        lr=1.0,
        max_seq_length=128,
        emb_dim=300,
        hidden_size=300,
        output_dim=2,
        vocab_size=30522,
        padding_idx=0,
        dropout_prob=0.5,
        save_steps=20,
        alpha=0.0,
        model_name='bert-base-uncased',
        teacher_path='model/SST-2/best_model_610/model_state.pdparams',
        vocab_path='/root/.paddlenlp/models/bert-base-uncased/bert-base-uncased-vocab.txt',
        use_pretrained_w2v=True,
        data_augmentation=False,
        n_iter=20):
    train_data_loader, dev_data_loader = create_distill_loader(
        task_name,
        language=language,
        model_name=model_name,
        vocab_path=vocab_path,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        data_augmentation=data_augmentation,
        n_iter=n_iter)

    emb_tensor = load_embedding(
        vocab_path=vocab_path) if use_pretrained_w2v else None
    model = BiLSTM(
        emb_dim,
        hidden_size,
        vocab_size,
        output_dim,
        padding_idx,
        dropout_prob=dropout_prob,
        embed_weight=emb_tensor)

    if opt == 'adadelta':
        optimizer = paddle.optimizer.Adadelta(
            learning_rate=lr, rho=0.95, parameters=model.parameters())
    else:
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=model.parameters())

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    klloss = nn.KLDivLoss()

    metric_class = TASK_CLASSES[task_name][1]
    metric = metric_class()

    teacher = Teacher(model_name=model_name, param_path=teacher_path)
    teacher_eval_logits_list = []
    with paddle.no_grad():
        for i, batch in enumerate(dev_data_loader):
            input_ids, segment_ids = batch[:2]
            teacher_logits = teacher.model(input_ids, segment_ids)
            teacher_eval_logits_list.append(teacher_logits)
    print("Start to train distilling model.")
    acc_list = []
    global_step = 0
    tic_train = time.time()
    for epoch in range(num_epoch):
        model.train()
        for i, batch in enumerate(train_data_loader):
            if task_name == 'qqp' or task_name == 'mnli':
                bert_input_ids, bert_segment_ids, small_input_ids_1, seq_len_1, small_input_ids_2, seq_len_2, labels = batch
            else:
                bert_input_ids, bert_segment_ids, small_input_ids, seq_len, labels = batch

            # Calculate teacher model's forward.
            with paddle.no_grad():
                teacher_logits = teacher.model(bert_input_ids, bert_segment_ids)

            # Calculate student model's forward.
            if task_name == 'qqp' or task_name == 'mnli':
                logits = model(small_input_ids_1, seq_len_1, small_input_ids_2,
                               seq_len_2)
            else:
                logits = model(small_input_ids, seq_len)

            loss = alpha * ce_loss(logits, labels) + (1 - alpha) * mse_loss(
                logits, teacher_logits)

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            if i % save_steps == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.4f step/s"
                    % (global_step, epoch, i, loss,
                       save_steps / (time.time() - tic_train)))
                tic_eval = time.time()
                if task_name == 'mnli':
                    evaluate(task_name, model, ce_loss, mse_loss, metric,
                             dev_data_loader_matched, teacher_eval_logits_list,
                             alpha)
                    evaluate(task_name, model, ce_loss, mse_loss, metric,
                             dev_data_loader_mismatched,
                             teacher_eval_logits_list, teacher_eval_logits_list,
                             alpha)
                    print("eval done total : %s s" % (time.time() - tic_eval))

                else:
                    acc = evaluate(task_name, model, ce_loss, mse_loss, metric,
                                   dev_data_loader, teacher_eval_logits_list,
                                   alpha)
                    print("eval done total : %s s" % (time.time() - tic_eval))
                    acc_list.append(acc)
                tic_train = time.time()
            global_step += 1

    # import matplotlib.pyplot as plt # 画图库
    # x_list = list(range(len(acc_list)))
    # x_list = [x * save_steps for x in x_list]
    # plt.plot(x_list, acc_list)
    # plt.ylabel('acc')
    # plt.xlabel('step')
    # plt.savefig(task_name+"_"+model_name+"_acc.png")


if __name__ == '__main__':
    paddle.seed(2021)
    # paddle.seed(202)
    # import numpy as np
    # task_name = 'senta'
    task_name = 'sst-2'
    # task_name = 'qqp'

    vocab_size_dict = {
        "senta": 29496,  #1256608,
        # "bert-base-chinese": 21128,
        # "bert-base-uncased": 30522,
        "sst-2": 30522,
        "qqp": 30522,
    }
    base_teacher_path_dict = {
        "sst-2": 'model/SST-2/best_model_610/model_state.pdparams',
        "qqp": "model/QQP/best_model_17000/model_state.pdparams",
        "mnli": "model/MNLI/best_model_18000/model_state.pdparams",
        "senta": 'model/chnsenticorp/best_model_930/model_state.pdparams'
    }
    large_teacher_path_dict = {
        "sst-2": 'model_large/SST-2/best_model_9450/model_state.pdparams',
    }
    if task_name == 'senta':
        do_train(
            task_name=task_name,
            alpha=0.0,
            dropout_prob=0.1,
            lr=1.0,
            language='cn',
            data_augmentation=True,
            batch_size=64,
            num_epoch=12,
            vocab_size=vocab_size_dict[task_name],
            model_name='bert-base-chinese',
            teacher_path=base_teacher_path_dict[task_name],
            use_pretrained_w2v=True,
            vocab_path='senta_word_dict_subset.txt')

    elif task_name == 'sst-2':
        do_train(
            task_name=task_name,
            alpha=0.0,
            dropout_prob=0.1,
            language='en',
            data_augmentation=True,
            batch_size=32,
            vocab_size=vocab_size_dict[task_name],
            model_name='bert-base-uncased',  # 'bert-base-uncased',
            teacher_path=base_teacher_path_dict[task_name],  # large_teacher
            use_pretrained_w2v=False)  # ???
    else:  # qqp
        do_train(
            task_name=task_name,
            alpha=0.0,
            dropout_prob=0.2,
            language='en',
            num_epoch=20,
            data_augmentation=True,
            batch_size=256,
            vocab_size=vocab_size_dict[task_name],
            model_name='bert-base-uncased',
            teacher_path=base_teacher_path_dict[task_name],
            use_pretrained_w2v=True,
            n_iter=10)
