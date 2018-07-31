
import tarfile
import gzip

import paddle.v2.dataset.common


__all__ = [
    'train',
    'test',
    'reader_creator',
    '__read_to_dict',
]



START = "<s>"
END = "<e>"
UNK = "<unk>"
UNK_IDX = 2


def __read_to_dict(file_name):

    with open(file_name,'r') as f:
        sentence = []
        lines = f.readlines()
        for line in lines:
            sentence.append(line.strip().split('\t')[-1])
            # print(sentence[-1])
        sentences = ' '.join(sentence)

        ll = sentences.split(' ')
        aa = list(set(ll))
        dic = {}
        aa = ['<s>'] + ['<e>'] + ["<unk>"] + aa
        for i in range(len(aa)):
            dic[aa[i]] = i
        
        return dic


def reader_creator(file_name):
    def reader():
        dic = __read_to_dict(file_name)
        
        with open(file_name, mode='r') as f:
            lines = f.readlines()
            for line in lines:

                line_split = line.strip().split('\t')
                if len(line_split) != 2:
                    print('!!!!!!')
                    continue

                image_path = line_split[0]
                trg_seq = line_split[1]  # one target sequence
                trg_words = trg_seq.split()
                trg_ids = [dic.get(w, UNK_IDX) for w in trg_words]

                # remove sequence whose length > 80 in training mode
                if len(trg_ids) > 80:
                    continue
                trg_ids_next = trg_ids + [dic[END]]
                trg_ids = [dic[START]] + trg_ids
                  
                b = proce(image_path)
                yield b, trg_ids, trg_ids_next

    return reader


def train():
    return reader_creator('data/train.txt')

def test():
    return reader_creator('data/test.txt')


def proce(img_path):
    img = paddle.v2.image.load_image(img_path)

    img = paddle.v2.image.simple_transform(img, 32, 32, True)

    image = img.flatten().astype('float32')
    return img
