#coding=utf-8

import cPickle


def concat_file(file1, file2, dst_file):
    with open(dst_file, 'w') as dst:
        with open(file1) as f1:
            with open(file2) as f2:
                for i, (line1, line2) in enumerate(zip(f1, f2)):
                    line1 = line1.strip()
                    line = line1 + '\t' + line2
                    dst.write(line)


if __name__ == '__main__':
    concat_file('dev.de-en.de', 'dev.de-en.en', 'dev')
    concat_file('test.de-en.de', 'test.de-en.en', 'test')
    concat_file('train.de-en.de', 'train.de-en.en', 'train')

    src_dict = cPickle.load(open('vocab.de'))
    trg_dict = cPickle.load(open('vocab.en'))

    with open('src_dict', 'w') as f:
        f.write('<s>\n<e>\nUNK\n')
        f.writelines('\n'.join(src_dict.keys()))

    with open('trg_dict', 'w') as f:
        f.write('<s>\n<e>\nUNK\n')
        f.writelines('\n'.join(trg_dict.keys()))
