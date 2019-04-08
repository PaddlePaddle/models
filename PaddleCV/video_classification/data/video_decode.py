import os, sys
import shutil


def decode():
    path = './UCF-101/'
    for folder in os.listdir(path):
        for vid in os.listdir(path + folder):
            print vid
            video_path = path + folder + '/' + vid
            image_folder = './frame/' + folder + '/' + vid.split('.')[0] + '/'
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)

            os.system('./ffmpeg -i ' + video_path + ' -q 0 ' + image_folder +
                      '/%06d.jpg')


if __name__ == '__main__':
    decode()
