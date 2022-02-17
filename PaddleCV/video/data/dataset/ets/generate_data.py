#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import json
import sys
sys.path.insert(0, '../../../coco-caption')
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def remove_nonascii(text):
    """ remove nonascii
    """
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


def generate_dictionary(caption_file_path):
    index = 0
    input_dict = {}

    # get all sentences
    train_data = json.loads(open(os.path.join( \
            caption_file_path, 'train.json')).read())
    for vid, content in train_data.iteritems():
        sentences = content['sentences']
        for s in sentences:
            input_dict[index] = [{'caption': remove_nonascii(s)}]
            index += 1

    # ptbtokenizer
    tokenizer = PTBTokenizer()
    output_dict = tokenizer.tokenize(input_dict)

    # sort by word frequency
    word_count_dict = {}
    for _, sentence in output_dict.iteritems():
        words = sentence[0].split()
        for w in words:
            if w not in word_count_dict:
                word_count_dict[w] = 1
            else:
                word_count_dict[w] += 1

    # output dictionary
    with open('dict.txt', 'w') as f:
        f.write('<s> -1\n')
        f.write('<e> -1\n')
        f.write('<unk> -1\n')

        truncation = 3
        for word, freq in sorted(word_count_dict.iteritems(), \
                key=lambda x:x[1], reverse=True):
            if freq >= truncation:
                f.write('%s %d\n' % (word, freq))

    print 'Generate dictionary done ...'


def generate_data_list(mode, caption_file_path):
    # get file name
    if mode == 'train':
        file_name = 'train.json'
    elif mode == 'val':
        file_name = 'val_1.json'
    else:
        print 'Invalid mode:' % mode
        sys.exit()

    # get timestamps and sentences
    input_dict = {}
    data = json.loads(open(os.path.join( \
            caption_file_path, file_name)).read())
    for vid, content in data.iteritems():
        sentences = content['sentences']
        timestamps = content['timestamps']
        for t, s in zip(timestamps, sentences):
            dictkey = ' '.join([vid, str(t[0]), str(t[1])])
            input_dict[dictkey] = [{'caption': remove_nonascii(s)}]

    # ptbtokenizer
    tokenizer = PTBTokenizer()
    output_dict = tokenizer.tokenize(input_dict)

    with open('%s.list' % mode, 'wb') as f:
        for id, sentence in output_dict.iteritems():
            try:
                f.write('\t'.join(id.split() + sentence) + '\n')
            except:
                pass

    print 'Generate %s.list done ...' % mode


if __name__ == '__main__':
    caption_file_path = './captions/'

    generate_dictionary(caption_file_path)

    generate_data_list('train', caption_file_path)
    generate_data_list('val', caption_file_path)
