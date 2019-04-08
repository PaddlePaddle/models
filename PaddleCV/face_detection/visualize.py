from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from PIL import Image
from PIL import ImageDraw


def draw_bbox(image, bbox):
    """
    Draw one bounding box on image.
    Args:
        image (PIL.Image): a PIL Image object.
        bbox (np.array|list|tuple): (xmin, ymin, xmax, ymax).
    """
    draw = ImageDraw.Draw(image)
    xmin, ymin, xmax, ymax = box
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top),
         (left, top)],
        width=4,
        fill='red')


def draw_bboxes(image_file, bboxes, labels=None, output_dir=None):
    """
    Draw bounding boxes on image.
    
    Args:
        image_file (string): input image path.
        bboxes (np.array): bounding boxes.
        labels (list of string): the label names of bboxes.
        output_dir (string): output directory.
    """
    if labels:
        assert len(bboxes) == len(labels)

    image = Image.open(image_file)
    draw = ImageDraw.Draw(image)
    for i in range(len(bboxes)):
        xmin, ymin, xmax, ymax = bboxes[i]
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        draw.line(
            [(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=4,
            fill='red')
        if labels and image.mode == 'RGB':
            draw.text((left, top), labels[i], (255, 255, 0))

    output_file = image_file.split('/')[-1]
    if output_dir:
        output_file = os.path.join(output_dir, output_file)

    print("The image with bbox is saved as {}".format(output_file))
    image.save(output_file)
