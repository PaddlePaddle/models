"""
operators to process sample, like decode/resize/crop image
"""
import uuid
import numpy as np
import cv2

class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def __call__(self, sample, context=None):
        """ process a sample

        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing

        Returns:
            result (dict): a processed sample
        """
        return sample

    def __str__(self):
        return '%s' % (self._id)


class DecodeImage(BaseOperator):
    def __init__(self, to_rgb=True, \
        to_np=False, channel_first=False):
        super(DecodeImage, self).__init__()
        self.to_rgb = to_rgb
        self.to_np = to_np  #to numpy
        self.channel_first = channel_first #only enabled when to_np is True

    def __call__(self, sample, context=None):
        assert 'image' in sample, 'not found image data'
        img = sample['image']

        data = np.frombuffer(img, dtype='uint8')
        img = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        sample['image'] = img
        return sample


class ResizeImage(BaseOperator):
    pass

