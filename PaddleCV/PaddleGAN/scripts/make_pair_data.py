import os
import argparse

parser = argparse.ArgumentParser(description='the direction of data list')
parser.add_argument(
    '--direction', type=str, default='B2A', help='the direction of data list')


def make_pair_data(fileA, file, d):
    f = open(fileA, 'r')
    lines = f.readlines()
    w = open(file, 'w')
    for line in lines:
        fileA = line[:-1]
        print(fileA)
        fileB = fileA.replace("A", "B")
        print(fileB)
        if d == 'A2B':
            l = fileA + '\t' + fileB + '\n'
        elif d == 'B2A':
            l = fileB + '\t' + fileA + '\n'
        else:
            raise NotImplementedError("the direction: [%s] is not support" % d)
        w.write(l)
    w.close()


if __name__ == "__main__":
    args = parser.parse_args()
    trainA_file = os.path.join("data", "cityscapes", "trainA.txt")
    train_file = os.path.join("data", "cityscapes", "pix2pix_train_list")
    make_pair_data(trainA_file, train_file, args.direction)

    testA_file = os.path.join("data", "cityscapes", "testA.txt")
    test_file = os.path.join("data", "cityscapes", "pix2pix_test_list")
    make_pair_data(testA_file, test_file, args.direction)
