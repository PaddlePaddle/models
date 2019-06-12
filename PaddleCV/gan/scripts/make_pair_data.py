import os


def make_pair_data(fileA, file):
    f = open(fileA, 'r')
    lines = f.readlines()
    w = open(file, 'w')
    for line in lines:
        fileA = line[:-1]
        print(fileA)
        fileB = fileA.replace("A", "B")
        print(fileB)
        l = fileA + '\t' + fileB + '\n'
        w.write(l)
    w.close()


if __name__ == "__main__":
    trainA_file = "./data/cityscapes/trainA.txt"
    train_file = "./data/cityscapes/pix2pix_train_list"
    make_pair_data(trainA_file, train_file)

    testA_file = "./data/cityscapes/testA.txt"
    test_file = "./data/cityscapes/pix2pix_test_list"
    make_pair_data(testA_file, test_file)
