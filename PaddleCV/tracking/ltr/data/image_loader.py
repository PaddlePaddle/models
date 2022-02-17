import jpeg4py
import cv2 as cv
import lmdb
import numpy as np


def default_image_loader(path):
    """The default image loader, reads the image from the given path. It first tries to use the jpeg4py_loader,
    but reverts to the opencv_loader if the former is not available."""
    if default_image_loader.use_jpeg4py is None:
        # Try using jpeg4py
        im = jpeg4py_loader(path)
        if im is None:
            default_image_loader.use_jpeg4py = False
            print('Jpeg4py is not available. Using OpenCV instead.')
        else:
            default_image_loader.use_jpeg4py = True
            return im
    if default_image_loader.use_jpeg4py:
        return jpeg4py_loader(path)
    return opencv_loader(path)


default_image_loader.use_jpeg4py = None


def jpeg4py_loader(path):
    """ Image reading using jpeg4py (https://github.com/ajkxyz/jpeg4py)"""
    try:
        return jpeg4py.JPEG(path).decode()
    except Exception as e:
        print('ERROR: Jpeg4py could not read image "{}". Using OpenCV instead.'.format(path))
        print(e)
        return opencv_loader(path)


def opencv_loader(path):
    """ Read image using opencv's imread function and returns it in rgb format"""
    try:
        im = cv.imread(path, cv.IMREAD_COLOR)
        # convert to rgb and return
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
    except Exception as e:
        print('ERROR: OpenCV could not read image "{}"'.format(path))
        print(e)
        return None


def lmdb_loader(path, lmdb_path=None):
    try:
        if lmdb_loader.txn is None:
            db = lmdb.open(lmdb_path, readonly=True, map_size=int(300e9))
            lmdb_loader.txn = db.begin(write=False)
        img_buffer = lmdb_loader.txn.get(path.encode())
        img_buffer = np.frombuffer(img_buffer, np.uint8)
        return cv.imdecode(img_buffer, cv.IMREAD_COLOR)
    except Exception as e:
        print('ERROR: Lmdb could not read image "{}"'.format(path))
        print(e)
        return None


lmdb_loader.txn = None
