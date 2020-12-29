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

    # def predict(self, text):
    #     tokens_raw = [self.tokenizer(l) for l in text] # example
    #     # Truncate to the truncate_length,
    #     tokens_trun = _truncate_seqs(tokens_raw, self.max_seq_len)
    #     # Concate the sequences with special tokens
    #     tokens_trun[0] = [self.tokenizer.cls_token] + tokens_trun[0]
    #     tokens, segment_ids, _ = _concat_seqs(tokens_trun, [[self.tokenizer.sep_token]] *
    #                                         len(tokens_trun))
    #     # Convert the token to ids
    #     input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    #     input_ids = paddle.to_tensor([input_ids])
    #     segment_ids = paddle.to_tensor([segment_ids])
    #     logits = self.model(input_ids, segment_ids)
    #     import pdb; pdb.set_trace()
    #     return F.softmax(logits).numpy()


def do_train(task_name='sst-2',
             num_epoch=20,
             batch_size=128,
             lr=0.5,
             max_seq_length=128,
             embed_dim=256,
             hidden_size=256,
             output_dim=2,
             vocab_size=30522,
             save_steps=100):
    alpha = 0.5
    train_data_loader, dev_data_loader = create_data_loader(
        task_name, batch_size, max_seq_length)

    model = BiLSTM(embed_dim, hidden_size, vocab_size, output_dim)

    gloabl_norm_clip = paddle.nn.ClipGradByNorm(5.0)
    adam = paddle.optimizer.Adam(
        learning_rate=lr,
        parameters=model.parameters(),
        grad_clip=gloabl_norm_clip)

    ce_loss = nn.CrossEntropyLoss()  #reduction='mean')
    mse_loss = nn.MSELoss()

    metric_class = TASK_CLASSES[task_name][1]
    metric = metric_class()

    global_step = 0
    tic_train = time.time()

    teacher_logits_list = []
    teacher = Teacher()
    with paddle.no_grad():
        for i, batch in enumerate(train_data_loader):
            input_ids, segment_ids, labels = batch
            teacher_logits = teacher.model(input_ids, segment_ids)
            teacher_logits_list.append(teacher_logits)
    del teacher

    for epoch in range(num_epoch):
        model.train()
        for i, batch in enumerate(train_data_loader):
            input_ids, segment_ids, labels = batch
            logits = model(input_ids[:, 1:])
            teacher_logits = teacher_logits_list[i]
            loss = alpha * ce_loss(logits, labels) + (1 - alpha) * mse_loss(
                F.softmax(logits), F.softmax(teacher_logits))

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
                    evaluate(model, loss_fct, metric, dev_data_loader_matched)
                    evaluate(model, loss_fct, metric,
                             dev_data_loader_mismatched)
                    print("eval done total : %s s" % (time.time() - tic_eval))

                else:
                    evaluate(model, loss_fct, metric, dev_data_loader)
                    print("eval done total : %s s" % (time.time() - tic_eval))
                tic_train = time.time()


if __name__ == '__main__':
    do_train()
