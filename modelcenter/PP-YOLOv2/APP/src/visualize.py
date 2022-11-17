import numpy as np
from PIL import Image, ImageDraw, ImageFile

def get_color_map_list(num_classes):
    """
    Args:
        num_classes (int): number of class
    Returns:
        color_map (list): RGB color list
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def draw_det(image, dt_bboxes, name_set):
    im = Image.fromarray(image)
    draw_thickness = min(im.size) // 320
    draw = ImageDraw.Draw(im)
    clsid2color = {}
    color_list = get_color_map_list(len(name_set))

    for (cls_id, score, xmin, ymin, xmax, ymax) in dt_bboxes:
        cls_id = int(cls_id)
        name = name_set[cls_id]
        color = tuple(color_list[cls_id])
        # draw bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=draw_thickness,
            fill=color)

        # draw label
        text = "{} {:.4f}".format(name, score)
        box = draw.textbbox((xmin, ymin), text, anchor='lt')
        draw.rectangle(box, fill=color)
        draw.text((box[0], box[1]), text, fill=(255, 255, 255))
    image = np.array(im)
    return image