# -*- coding: utf-8 -*-
import os
import io
import re
import json
import click
import collections


def build_vocabulary(dataset, cutoff=0):
    dictionary = collections.defaultdict(int)
    for data in dataset:
        for sent in data[2]:
            for char in sent:
                dictionary[char] += 1
    dictionary = filter(lambda x: x[1] >= cutoff, dictionary.items())
    dictionary = sorted(dictionary, key=lambda x: (-x[1], x[0]))
    vocab, _ = list(zip(*dictionary))
    return (u"<s>", u"<e>", u"<unk>") + vocab


@click.command("preprocess")
@click.option("--datadir", type=str, help="Path to raw data")
@click.option("--outfile", type=str, help="Path to save the training data")
@click.option("--dictfile", type=str, help="Path to save the dictionary file")
def preprocess(datadir, outfile, dictfile):
    dataset = []
    note_pattern1 = re.compile(u"（.*?）", re.U)
    note_pattern2 = re.compile(u"〖.*?〗", re.U)
    note_pattern3 = re.compile(u"-.*?-。?", re.U)
    note_pattern4 = re.compile(u"（.*$", re.U)
    note_pattern5 = re.compile(u"。。.*）$", re.U)
    note_pattern6 = re.compile(u"。。", re.U)
    note_pattern7 = re.compile(u"[《》「」\[\]]", re.U)
    print("Load raw data...")
    for fn in os.listdir(datadir):
        with io.open(os.path.join(datadir, fn), "r", encoding="utf8") as f:
            for data in json.load(f):
                title = data['title']
                author = data['author']
                p = "".join(data['paragraphs'])
                p = "".join(p.split())
                p = note_pattern1.sub(u"", p)
                p = note_pattern2.sub(u"", p)
                p = note_pattern3.sub(u"", p)
                p = note_pattern4.sub(u"", p)
                p = note_pattern5.sub(u"。", p)
                p = note_pattern6.sub(u"。", p)
                p = note_pattern7.sub(u"", p)
                if (p == u"" or u"{" in p or u"}" in p or u"｛" in p or
                        u"｝" in p or u"、" in p or u"：" in p or u"；" in p or
                        u"！" in p or u"？" in p or u"●" in p or u"□" in p or
                        u"囗" in p or u"）" in p):
                    continue
                paragraphs = re.split(u"。|，", p)
                paragraphs = filter(lambda x: len(x), paragraphs)
                if len(paragraphs) > 1:
                    dataset.append((title, author, paragraphs))

    print("Construct vocabularies...")
    vocab = build_vocabulary(dataset, cutoff=10)
    with io.open(dictfile, "w", encoding="utf8") as f:
        for v in vocab:
            f.write(v + "\n")

    print("Write processed data...")
    with io.open(outfile, "w", encoding="utf8") as f:
        for data in dataset:
            title = data[0]
            author = data[1]
            paragraphs = ".".join(data[2])
            f.write("\t".join((title, author, paragraphs)) + "\n")


if __name__ == "__main__":
    preprocess()
