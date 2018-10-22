from utils import load_dict


def train_reader(data_file_path, word_dict_file):
    def reader():
        word_dict = load_dict(word_dict_file)

        unk_id = word_dict[u"<unk>"]
        bos_id = word_dict[u"<s>"]
        eos_id = word_dict[u"<e>"]

        with open(data_file_path, "r") as f:
            for line in f:
                line_split = line.strip().decode(
                    "utf8", errors="ignore").split("\t")
                if len(line_split) < 3: continue

                poetry = line_split[2].split(".")
                poetry_ids = []
                for sen in poetry:
                    if sen:
                        poetry_ids.append([bos_id] + [
                            word_dict.get(word, unk_id)
                            for word in "".join(sen.split())
                        ] + [eos_id])
                l = len(poetry_ids)
                if l < 2: continue
                for i in range(l - 1):
                    yield poetry_ids[i], poetry_ids[i + 1][:-1], poetry_ids[
                        i + 1][1:]

    return reader


def gen_reader(data_file_path, word_dict_file):
    def reader():
        word_dict = load_dict(word_dict_file)

        unk_id = word_dict[u"<unk>"]
        bos_id = word_dict[u"<s>"]
        eos_id = word_dict[u"<e>"]

        with open(data_file_path, "r") as f:
            for line in f:
                input_line = "".join(line.strip().decode(
                    "utf8", errors="ignore").split())
                yield [bos_id] + [
                    word_dict.get(word, unk_id) for word in input_line
                ] + [eos_id]

    return reader
