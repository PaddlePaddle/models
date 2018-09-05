from PIL import Image
import numpy as np

A_LIST_FILE="./data/horse2zebra/trainA.txt"
B_LIST_FILE="./data/horse2zebra/trainB.txt"
IMAGES_ROOT="./data/horse2zebra/"

def reader_creater(list_file, cycle=False):
    images = [IMAGES_ROOT+line for line in open(list_file, 'r').readlines()]
    
    def reader():
        while True:
            np.random.shuffle(images)
            for file in images:
                image = Image.open(file.strip("\n\r\t "))
                image = image.resize((256, 256))
                image = np.array(image) / 127.5 - 1
                image = image.transpose([2, 0, 1])
                yield image
            if not cycle:
                break
    return reader

def a_reader():
    return reader_creater(A_LIST_FILE)

def b_reader():
    return reader_creater(B_LIST_FILE)

