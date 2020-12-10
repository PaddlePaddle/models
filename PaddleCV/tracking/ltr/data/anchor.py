import math
import numpy as np
from collections import namedtuple

Corner = namedtuple('Corner', 'x1 y1 x2 y2')

# alias
BBox = Corner
Center = namedtuple('Center', 'x y w h')

def topleft2corner(topleft):
    """ convert (x, y, w, h) to (x1, y1, x2, y2)
    Args:
        center: np.array (4 * N)
    Return:
        np.array (4 * N)
    """
    x, y, w, h = topleft[0], topleft[1], topleft[2], topleft[3]
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return x1, y1, x2, y2

def corner2center(corner):
    """ convert (x1, y1, x2, y2) to (cx, cy, w, h)
    Args:
        conrner: Corner or np.array (4*N)
    Return:
        Center or np.array (4 * N)
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h


def center2corner(center):
    """ convert (cx, cy, w, h) to (x1, y1, x2, y2)
    Args:
        center: Center or np.array (4 * N)
    Return:
        center or np.array (4 * N)
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2


def IoU(rect1, rect2):
    """ caculate interection over union
    Args:
        rect1: (x1, y1, x2, y2)
        rect2: (x1, y1, x2, y2)
    Returns:
        iou
    """
    # overlap
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)

    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)

    area = (x2 - x1) * (y2 - y1)
    target_a = (tx2 - tx1) * (ty2 - ty1)
    inter = ww * hh
    iou = inter / (area + target_a - inter)
    return iou


class Anchors:
    """
    This class generate anchors.
    """

    def __init__(self, stride, ratios, scales, image_center=0, size=0):
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.image_center = 0
        self.size = 0

        self.anchor_num = len(self.scales) * len(self.ratios)

        self.anchors = None

        self.generate_anchors()

    def generate_anchors(self):
        """
        generate anchors based on predefined configuration
        """
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride * self.stride
        count = 0
        for r in self.ratios:
            ws = int(math.sqrt(size * 1. / r))
            hs = int(ws * r)

            for s in self.scales:
                w = ws * s
                h = hs * s
                self.anchors[count][:] = [-w * 0.5, -h * 0.5, w * 0.5, h * 0.5][:]
                count += 1

    def generate_all_anchors(self, im_c, size):
        """
        im_c: image center
        size: image size
        """
        if self.image_center == im_c and self.size == size:
            return False
        self.image_center = im_c
        self.size = size

        a0x = im_c - size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero_anchors = self.anchors + ori

        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]

        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1),
                             [x1, y1, x2, y2])
        cx, cy, w, h = corner2center([x1, y1, x2, y2])

        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride

        cx = cx + disp_x
        cy = cy + disp_y

        # broadcast
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])

        self.all_anchors = (np.stack([x1, y1, x2, y2]).astype(np.float32),
                            np.stack([cx, cy, w, h]).astype(np.float32))
        return True


class AnchorTarget:
    def __init__(self,
                 search_size,
                 output_size,
                 stride,
                 ratios,
                 scales,
                 num_pos,
                 num_neg,
                 num_total,
                 thr_high,
                 thr_low):
        self.search_size = search_size
        self.output_size = output_size
        self.anchor_stride = stride
        self.anchor_ratios = ratios
        self.anchor_scales = scales
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.num_total = num_total
        self.thr_high = thr_high
        self.thr_low = thr_low

        self.anchors = Anchors(stride,
                               ratios,
                               scales)

        self.anchors.generate_all_anchors(im_c=search_size // 2,
                                          size=output_size)

    def __call__(self, target, size, neg=False):
        anchor_num = len(self.anchor_ratios) * len(self.anchor_scales)

        # -1 ignore 0 negative 1 positive
        cls = -1 * np.ones((anchor_num, size, size), dtype=np.int64)
        delta = np.zeros((4, anchor_num, size, size), dtype=np.float32)
        delta_weight = np.zeros((anchor_num, size, size), dtype=np.float32)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        tcx, tcy, tw, th = corner2center(target)

        if neg:
            # l = size // 2 - 3
            # r = size // 2 + 3 + 1
            # cls[:, l:r, l:r] = 0

            cx = size // 2
            cy = size // 2
            cx += int(np.ceil((tcx - self.search_size // 2) /
                              self.anchor_stride + 0.5))
            cy += int(np.ceil((tcy - self.search_size // 2) /
                              self.anchor_stride + 0.5))
            l = max(0, cx - 3)
            r = min(size, cx + 4)
            u = max(0, cy - 3)
            d = min(size, cy + 4)
            cls[:, u:d, l:r] = 0

            neg, neg_num = select(np.where(cls == 0), self.num_neg)
            cls[:] = -1
            cls[neg] = 0

            overlap = np.zeros((anchor_num, size, size), dtype=np.float32)
            return cls, delta, delta_weight, overlap

        anchor_box = self.anchors.all_anchors[0]
        anchor_center = self.anchors.all_anchors[1]
        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], \
                         anchor_box[2], anchor_box[3]
        cx, cy, w, h = anchor_center[0], anchor_center[1], \
                       anchor_center[2], anchor_center[3]

        delta[0] = (tcx - cx) / w
        delta[1] = (tcy - cy) / h
        delta[2] = np.log(tw / w)
        delta[3] = np.log(th / h)

        overlap = IoU([x1, y1, x2, y2], target)

        pos = np.where(overlap > self.thr_high)
        neg = np.where(overlap < self.thr_low)

        pos, pos_num = select(pos, self.num_pos)
        neg, neg_num = select(neg, self.num_total - self.num_pos)

        cls[pos] = 1
        delta_weight[pos] = 1. / (pos_num + 1e-6)

        cls[neg] = 0
        return cls, delta, delta_weight, overlap
