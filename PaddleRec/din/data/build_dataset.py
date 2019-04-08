from __future__ import print_function
import random
import pickle

random.seed(1234)

print("read and process data")

with open('./raw_data/remap.pkl', 'rb') as f:
    reviews_df = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count, example_count = pickle.load(f)

train_set = []
test_set = []
for reviewerID, hist in reviews_df.groupby('reviewerID'):
    pos_list = hist['asin'].tolist()

    def gen_neg():
        neg = pos_list[0]
        while neg in pos_list:
            neg = random.randint(0, item_count - 1)
        return neg

    neg_list = [gen_neg() for i in range(len(pos_list))]

    for i in range(1, len(pos_list)):
        hist = pos_list[:i]
        if i != len(pos_list) - 1:
            train_set.append((reviewerID, hist, pos_list[i], 1))
            train_set.append((reviewerID, hist, neg_list[i], 0))
        else:
            label = (pos_list[i], neg_list[i])
            test_set.append((reviewerID, hist, label))

random.shuffle(train_set)
random.shuffle(test_set)

assert len(test_set) == user_count


def print_to_file(data, fout):
    for i in range(len(data)):
        fout.write(str(data[i]))
        if i != len(data) - 1:
            fout.write(' ')
        else:
            fout.write(';')


print("make train data")
with open("paddle_train.txt", "w") as fout:
    for line in train_set:
        history = line[1]
        target = line[2]
        label = line[3]
        cate = [cate_list[x] for x in history]
        print_to_file(history, fout)
        print_to_file(cate, fout)
        fout.write(str(target) + ";")
        fout.write(str(cate_list[target]) + ";")
        fout.write(str(label) + "\n")

print("make test data")
with open("paddle_test.txt", "w") as fout:
    for line in test_set:
        history = line[1]
        target = line[2]
        cate = [cate_list[x] for x in history]

        print_to_file(history, fout)
        print_to_file(cate, fout)
        fout.write(str(target[0]) + ";")
        fout.write(str(cate_list[target[0]]) + ";")
        fout.write("1\n")

        print_to_file(history, fout)
        print_to_file(cate, fout)
        fout.write(str(target[1]) + ";")
        fout.write(str(cate_list[target[1]]) + ";")
        fout.write("0\n")

print("make config data")
with open('config.txt', 'w') as f:
    f.write(str(user_count) + "\n")
    f.write(str(item_count) + "\n")
    f.write(str(cate_count) + "\n")
