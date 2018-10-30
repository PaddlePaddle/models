#coding=utf-8
"""
Convert the raw json data into training and validation examples.
"""
from collections import Counter
import json
import os
import io
import string

import click
import numpy as np
import ciseau

from vocab import Vocab
from evaluate import normalize_answer

# Constants
UNK = "<UNK>"
SOS = "<SOS>"
EOS = "<EOS>"
PAD = "<PAD>"

splits = ["train", "dev"]

ARTICLES = {"a", "an", "the", "of"}

# Keep the random embedding matrix the same between runs.
np.random.seed(1234)


def data_stream(path):
    """ Given a path json data in Pranav format, convert it to a stream
    question/context/answers tuple."""
    with io.open(path, "r") as handle:
        raw_data = json.load(handle)["data"]
    for ex in raw_data:
        for paragraph in ex["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                answers = qa["answers"]
                if "id" not in qa:
                    qa_id = -1
                else:
                    qa_id = qa["id"]
                yield question, context, answers, qa_id


def build_vocabulary(datadir, outdir, glove_path):
    """Construct the vocabulary object used throughout."""
    # We're not going to backprop through the word vectors
    # both train and dev words end up in the vocab.
    counter = Counter()
    for split in splits:
        datapath = os.path.join(datadir, split + ".json")

        for question, context, _, _ in data_stream(datapath):
            for word in ciseau.tokenize(question, normalize_ascii=False):
                counter[normalize(word)] += 1
            for word in ciseau.tokenize(context, normalize_ascii=False):
                counter[normalize(word)] += 1

    common_words = [UNK, SOS, EOS, PAD] + [w for w, _ in counter.most_common()]

    vocab_path = os.path.join(outdir, "vocab.txt")
    with io.open(vocab_path, "w", encoding="utf8") as handle:
        handle.write("\n".join(common_words))

    return Vocab(outdir)


def normalize_answer_tokens(tokens):
    start = 0
    end = len(tokens)

    while end - start > 1:
        first_token = tokens[start].rstrip().lower()
        if first_token in string.punctuation or first_token in ARTICLES:
            start += 1
        else:
            break
    while end - start > 1:
        last_token = tokens[end - 1].rstrip().lower()
        if last_token in string.punctuation:
            end -= 1
        else:
            break
    return start, end


def tokenize_example(question, context, answers, strip_labels=True):
    # Q: How should we choose the right answer
    answer = answers[0]["text"]
    answer_start = answers[0]["answer_start"]

    if strip_labels:
        answer_tokens = ciseau.tokenize(answer, normalize_ascii=False)
        start_offset, end_offset = normalize_answer_tokens(answer_tokens)
        answer = "".join(answer_tokens[start_offset:end_offset])
        # add back the piece that was stripped off:
        answer_start = answer_start + len("".join(answer_tokens[:start_offset]))

    # replace answer string with placeholder
    placeholder = "XXXX"
    new_context = context[:answer_start] + placeholder + context[answer_start +
                                                                 len(answer):]

    token_context = ciseau.sent_tokenize(new_context, keep_whitespace=True)
    token_question = ciseau.tokenize(question)

    sentence_label = None
    for sent_idx, sent in enumerate(token_context):
        answer_start = None
        for idx, word in enumerate(sent):
            if placeholder in word:
                answer_start = idx
                break

        if answer_start is None:
            continue

        sentence_label = sent_idx

        # deal with cases where the answer is in the middle
        # of the word
        answer = word.replace(placeholder, answer)
        token_answer = ciseau.tokenize(answer)

        answer_end = answer_start + len(token_answer) - 1
        answer_sent = sent[:answer_start] + token_answer + sent[answer_start +
                                                                1:]
        break

    token_context[sentence_label] = answer_sent

    return token_question, token_context, sentence_label, answer_start, answer_end


def normalize(word):
    return word.strip()


def same_as_question_feature(question_idxs, context_idxs, vocab):
    question_words = [vocab.idx_to_word(idx) for idx in question_idxs]

    # remove stop word and puncutation
    question_words = set([
        w.strip().lower() for w in question_words
        if w not in ARTICLES and w not in string.punctuation
    ])

    features = []
    for word_idx in context_idxs:
        word = vocab.idx_to_word(word_idx)
        features.append(int(word.strip().lower() in question_words))

    return features


def repeated_word_features(context_idxs, vocab):
    context_words = [vocab.idx_to_word(idx) for idx in context_idxs]

    word_counter = {}
    for word in context_words:
        canon = word.strip().lower()
        if canon in word_counter:
            word_counter[canon] += 1
        else:
            word_counter[canon] = 1

    max_occur = max(word_counter.values())
    min_occur = min(word_counter.values())
    occur_range = max(1.0, max_occur - min_occur)

    repeated_words = []
    repeated_word_intensity = []

    for word in context_words:
        canon = word.strip().lower()
        count = word_counter[canon]
        repeated = float(count > 1 and canon not in ARTICLES and
                         canon not in string.punctuation)
        intensity = float((count - min_occur) / occur_range)

        repeated_words.append(repeated)
        repeated_word_intensity.append(intensity)

    return repeated_words, repeated_word_intensity


def convert_example_to_indices(example, outfile, vocab):
    print("Processing {}".format(outfile))
    question, context, answers, qa_id = example

    tokenized = tokenize_example(question, context, answers, strip_labels=True)
    token_question, token_context, ans_sent, ans_start, ans_end = tokenized

    # Convert to indices
    question_idxs = [vocab.word_to_idx(normalize(w)) for w in token_question]

    # + 1 for end of sentence
    sent_lengths = [len(sent) + 1 for sent in token_context]
    context_idxs = []
    for sent in token_context:
        for w in sent:
            context_idxs.append(vocab.word_to_idx(normalize(w)))
        context_idxs.append(vocab.eos)

    same_as_question = same_as_question_feature(question_idxs, context_idxs,
                                                vocab)

    repeated_words, repeated_intensity = repeated_word_features(context_idxs,
                                                                vocab)

    features = {
        "question": question_idxs,
        "context": context_idxs,
        "ans_sentence": ans_sent,
        "ans_start": ans_start,
        "ans_end": ans_end,
        "sent_lengths": sent_lengths,
        "same_as_question_word": same_as_question,
        "repeated_words": repeated_words,
        "repeated_intensity": repeated_intensity,
        "qa_id": qa_id
    }

    # Hack!: This is not a great way to save indices...
    with io.open(outfile, "w", encoding="utf8") as handle:
        handle.write(unicode(json.dumps(features, ensure_ascii=False)))


def featurize_example(question, context, vocab):
    # Convert to indices
    question_idxs = [
        vocab.word_to_idx(normalize(w))
        for w in ciseau.tokenize(
            question, normalize_ascii=False)
    ]

    context_sents = ciseau.sent_tokenize(
        context, keep_whitespace=True, normalize_ascii=False)
    # + 1 for end of sentence
    sent_lengths = [len(sent) + 1 for sent in context_sents]
    context_idxs = []
    for sent in context_sents:
        for w in sent:
            context_idxs.append(vocab.word_to_idx(normalize(w)))
        context_idxs.append(vocab.eos)

    same_as_question = same_as_question_feature(question_idxs, context_idxs,
                                                vocab)
    repeated_words, repeated_intensity = repeated_word_features(context_idxs,
                                                                vocab)

    return (question_idxs, context_idxs, same_as_question, repeated_words,
            repeated_intensity, sent_lengths), context_sents


def random_sample(data, k, replace=False):
    indices = np.arange(len(data))
    chosen_indices = np.random.choice(indices, k, replace=replace)
    return [data[idx] for idx in chosen_indices]


@click.command()
@click.option("--datadir", type=str, help="Path to raw data")
@click.option("--outdir", type=str, help="Path to save the result")
@click.option("--glove-path", default="/mnt/data/jmiller/glove.840B.300d.txt")
def preprocess(datadir, outdir, glove_path):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Constructing vocabularies...")
    vocab = build_vocabulary(datadir, outdir, glove_path)
    print("Finished...")

    print("Building word embedding matrix...")
    vocab.construct_embedding_matrix(glove_path)
    print("Finished...")

    # Create training featurizations
    for split in splits:
        results_path = os.path.join(outdir, split)
        os.makedirs(results_path)

        # process each example
        examples = list(data_stream(os.path.join(datadir, split + ".json")))

        for idx, example in enumerate(examples):
            outfile = os.path.join(results_path, str(idx) + ".json")
            convert_example_to_indices(example, outfile, vocab)

    print("Building evaluation featurization...")
    eval_feats = []
    for question, context, _, qa_id in data_stream(
            os.path.join(datadir, "dev.json")):
        features, tokenized_context = featurize_example(question, context,
                                                        vocab)
        eval_feats.append((qa_id, tokenized_context, features))

    with io.open(
            os.path.join(outdir, "eval.json"), "w", encoding="utf8") as handle:
        handle.write(unicode(json.dumps(eval_feats, ensure_ascii=False)))


if __name__ == "__main__":
    preprocess()
