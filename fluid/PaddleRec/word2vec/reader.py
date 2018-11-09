# -*- coding: utf-8 -*

import time
import numpy as np
import random
from collections import Counter
"""
refs: https://github.com/NELSONZHAO/zhihu/blob/master/skip_gram/Skip-Gram-English-Corpus.ipynb
"""

with open('data/text8.txt') as f:
    text = f.read()


# 定义函数来完成数据的预处理
def preprocess(text, freq=5):
    '''
    对文本进行预处理

    参数
    ---
    text: 文本数据
    freq: 词频阈值
    '''
    # 对文本中的符号进行替换
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()

    # 删除低频词，减少噪音影响
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > freq]

    return trimmed_words


# 清洗文本并分词
words = preprocess(text)
print(words[:20])

# 构建映射表
vocab = set(words)
vocab_to_int = {w: c for c, w in enumerate(vocab)}
int_to_vocab = {c: w for c, w in enumerate(vocab)}

dict_size = len(set(words))

print("total words: {}".format(len(words)))
print("unique words: {}".format(dict_size))

# 对原文本进行vocab到int的转换
int_words = [vocab_to_int[w] for w in words]

t = 1e-5  # t值
threshold = 0.8  # 剔除概率阈值

# # 统计单词出现频次
# int_word_counts = Counter(int_words)
# total_count = len(int_words)
# # 计算单词频率
# word_freqs = {w: c/total_count for w, c in int_word_counts.items()}
# # 计算被删除的概率
# prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in int_word_counts}
# # 对单词进行采样
# train_words = [w for w in int_words if prob_drop[w] < threshold]

train_words = int_words
len(train_words)


def get_targets(words, idx, window_size=5):
    '''
    获得input word的上下文单词列表

    参数
    ---
    words: 单词列表
    idx: input word的索引号
    window_size: 窗口大小
    '''
    target_window = np.random.randint(1, window_size + 1)
    # 这里要考虑input word前面单词不够的情况
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    # output words(即窗口中的上下文单词)
    targets = set(words[start_point:idx] + words[idx + 1:end_point + 1])
    return list(targets)


def get_batches(words, batch_size, window_size=5):
    def _reader():
        '''
        构造一个获取batch的生成器
        '''
        n_batches = len(words) // batch_size

        # 仅取full batches
        new_words = words[:n_batches * batch_size]

        for idx in range(0, len(new_words), batch_size):
            x, y = [], []
            batch = new_words[idx:idx + batch_size]
            for i in range(len(batch)):
                batch_x = batch[i]
                batch_y = get_targets(batch, i, window_size)
                # 由于一个input word会对应多个output word，因此需要长度统一
                x.extend([batch_x] * len(batch_y))
                y.extend(batch_y)
            for i in range(len(batch_y)):
                yield [x[i]], [y[i]]

    return _reader


if __name__ == "__main__":
    epochs = 10  # 迭代轮数
    batch_size = 1000  # batch大小
    window_size = 10  # 窗口大小

    batches = get_batches(train_words, batch_size, window_size)
    i = 0
    for x, y in batches():
        print("x: " + str(x))
        print("y: " + str(y))
        print("\n")
        if i == 10:
            exit(0)
        i += 1
