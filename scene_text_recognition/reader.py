import os
import cv2

from paddle.v2.image import load_image


class DataGenerator(object):
    def __init__(self, char_dict, image_shape):
        '''
        :param char_dict: The dictionary class for labels.
        :type char_dict: class
        :param image_shape: The fixed shape of images.
        :type image_shape: tuple
        '''
        self.image_shape = image_shape
        self.char_dict = char_dict

    def train_reader(self, file_list):
        '''
        Reader interface for training.
        
        :param file_list: The path list of the image file for training.
        :type file_list: list
        '''

        def reader():
            for i, (image, label) in enumerate(file_list):
                yield self.load_image(image), self.char_dict.word2ids(label)

        return reader

    def infer_reader(self, file_list):
        '''
        Reader interface for inference.
           
        :param file_list: The path list of the image file for inference.
        :type file_list: list
        '''

        def reader():
            for i, (image, label) in enumerate(file_list):
                yield self.load_image(image), label

        return reader

    def load_image(self, path):
        '''
        Load image and transform to 1-dimention vector.
           
        :param path: The path of the image data.
        :type path: str
        '''
        image = load_image(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize all images to a fixed shape.
        if self.image_shape:
            image = cv2.resize(
                image, self.image_shape, interpolation=cv2.INTER_CUBIC)

        image = image.flatten() / 255.
        return image
