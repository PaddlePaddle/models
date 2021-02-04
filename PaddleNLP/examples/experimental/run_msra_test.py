from datasets import MSRA_NER
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.data import Stack, Tuple, Pad, Dict
from functools import partial
from paddle.io import DataLoader

train_ds, test_ds = MSRA_NER().read_datasets('train', 'test')

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

ignore_label = -100


def tokenize_and_align_labels(examples, no_entity_id):
    labels = [examples[i]['labels'] for i in range(len(examples))]
    examples = [examples[i]['tokens'] for i in range(len(examples))]
    tokenized_inputs = tokenizer(
        examples,
        is_split_into_words=True,
        max_seq_len=512,
        pad_to_max_seq_len=True)

    for i in range(len(tokenized_inputs)):
        tokenized_inputs[i]['labels'] = [no_entity_id] + labels[
            i] + [no_entity_id]
        tokenized_inputs[i]['labels'] += [ignore_label] * (
            len(tokenized_inputs[i]['input_ids']) -
            len(tokenized_inputs[i]['labels']))
    return tokenized_inputs


label_list = MSRA_NER().get_labels()
no_entity_id = len(label_list) - 1

trans_func = partial(tokenize_and_align_labels, no_entity_id=no_entity_id)

train_ds.map(trans_func)

print(train_ds[0])
print(len(train_ds))
print('-----------------------------------------------------')

batchify_fn = lambda samples, fn=Dict({
    'input_ids': Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # input
    'segment_ids': Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # segment
    'labels': Pad(axis=0, pad_val=ignore_label)  # label
}): fn(samples)

train_data_loader = DataLoader(
    dataset=train_ds,
    batch_size=8,
    collate_fn=batchify_fn,
    num_workers=0,
    return_list=True)

for batch in train_data_loader:
    print(batch[0])
    print(batch[1])
    print(batch[2])
    break