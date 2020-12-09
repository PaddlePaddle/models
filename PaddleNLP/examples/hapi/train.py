from functools import partial

from paddle.io import DistributedBatchSampler, DataLoader
from paddle.static import InputSpec
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer, ErnieForSequenceClassification, ErnieTokenizer
import numpy as np
import paddle
import paddlenlp


def convert_example(example, tokenizer, max_seq_length=128):
    text, label = example
    encoded_inputs = tokenizer.encode(text, max_seq_len=max_seq_length)
    input_ids, segment_ids = encoded_inputs["input_ids"], encoded_inputs["segment_ids"]
    label = np.array([label], dtype="int64")
    return input_ids, segment_ids, label

paddle.set_device('gpu')
train_ds = paddlenlp.datasets.ChnSentiCorp.get_datasets(['train'])
label_list = train_ds.get_labels()
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
trans_func = partial(convert_example, tokenizer=tokenizer)
train_ds = train_ds.apply(trans_func)
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),
    Pad(axis=0, pad_val=tokenizer.pad_token_id),
    Stack(dtype="int64") ): [data for data in fn(samples)]
batch_sampler = DistributedBatchSampler(train_ds, batch_size=32, shuffle=True)
train_loader = DataLoader(
    dataset=train_ds,
    batch_sampler=batch_sampler,
    collate_fn=batchify_fn,
    return_list=True)

model = paddlenlp.models.Ernie('ernie-1.0', task='seq-cls', num_classes=len(label_list))
criterion = paddle.nn.loss.CrossEntropyLoss()
metric = paddle.metric.Accuracy()
optimizer = paddle.optimizer.AdamW(
    learning_rate=5e-5, parameters=model.parameters())

inputs = [
    InputSpec([None, 128], dtype='int64', name='input_ids'), 
    InputSpec([None, 128], dtype='int64', name='token_type_ids')
]
trainer = paddle.Model(model, inputs)
trainer.prepare(optimizer, criterion, metric)
trainer.fit(train_loader, batch_size=32, epochs=3)
