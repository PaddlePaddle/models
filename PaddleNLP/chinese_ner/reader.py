import os


def file_reader(file_dir):
    def reader():
        files = os.listdir(file_dir)
        for fi in files:
            for line in open(file_dir + '/' + fi, 'r'):
                line = line.strip()
                features = line.split(";")
                word_idx = []
                for item in features[1].strip().split(" "):
                    word_idx.append(int(item))
                    target_idx = []
                for item in features[2].strip().split(" "):
                    label_index = int(item)
                    if label_index == 0:
                        label_index = 48
                    else:
                        label_index -= 1
                    target_idx.append(label_index)
                mention_idx = []
                for item in features[3].strip().split(" "):
                    mention_idx.append(int(item))
                yield word_idx, mention_idx, target_idx,

    return reader
