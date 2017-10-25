import os


class AsciiDic(object):
    UNK_ID = 0

    def __init__(self):
        self.dic = {
            '<unk>': self.UNK_ID,
        }
        self.chars = [chr(i) for i in range(40, 171)]
        for id, c in enumerate(self.chars):
            self.dic[c] = id + 1

    def lookup(self, w):
        return self.dic.get(w, self.UNK_ID)

    def id2word(self):
        '''
        Return a reversed char dict.
        '''
        self.id2word = {}
        for key, value in self.dic.items():
            self.id2word[value] = key

        return self.id2word

    def word2ids(self, word):
        '''
        Transform a word to a list of ids.

        :param word: The word appears in image data.
        :type word: str
        '''
        return [self.lookup(c) for c in list(word)]

    def size(self):
        return len(self.dic)


def get_file_list(image_file_list):
    '''
    Generate the file list for training and testing data.
    
    :param image_file_list: The path of the file which contains
                           path list of image files.
    :type image_file_list: str
    '''
    dirname = os.path.dirname(image_file_list)
    path_list = []
    with open(image_file_list) as f:
        for line in f:
            line_split = line.strip().split(',', 1)
            filename = line_split[0].strip()
            path = os.path.join(dirname, filename)
            label = line_split[1][2:-1]
            path_list.append((path, label))

    return path_list
