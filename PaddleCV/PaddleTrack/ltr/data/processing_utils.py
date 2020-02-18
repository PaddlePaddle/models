import math
import numpy as np
import cv2 as cv

def stack_tensors(x):
    if isinstance(x, list) and isinstance(x[0], np.ndarray):
        return np.stack(x)
    return x


def sample_target(im, target_bb, search_area_factor, output_sz=None,
                  scale_type='original', border_type='replicate'):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """

    x, y, w, h = target_bb.tolist()

    # Crop image
    if scale_type == 'original':
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    elif scale_type == 'context':
        # some context is added into the target_size
        # now, the search factor is respect to the "target + context"
        # when search_factor = 1, output_size = 127
        # when search_factor = 2, output_size = 255
        context = (w + h) / 2
        base_size = math.sqrt((w + context) * (h + context))  # corresponds to 127 in crop
        crop_sz = math.ceil(search_area_factor * base_size)
    else:
        raise NotImplementedError

    if crop_sz < 1:
        raise Exception('Too small bounding box. w: {}, h: {}'.format(w, h))

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    # Pad
    if border_type == 'replicate':
        im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_REPLICATE)
    elif border_type == 'zeropad':
        im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    elif border_type == 'meanpad':
        avg_chans = np.array([np.mean(im[:, :, 0]), np.mean(im[:, :, 1]), np.mean(im[:, :, 2])])
        im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT, value=avg_chans)
    else:
        raise NotImplementedError

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        return cv.resize(im_crop_padded, (output_sz, output_sz)), resize_factor
    else:
        return im_crop_padded, 1.0


def transform_image_to_crop(box_in: np.ndarray, box_extract: np.ndarray, resize_factor: float,
                            crop_sz: np.ndarray) -> np.ndarray:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = np.concatenate((box_out_center - 0.5 * box_out_wh, box_out_wh))
    return box_out


def centered_crop(frames, anno, area_factor, output_sz):
    crops_resize_factors = [sample_target(f, a, area_factor, output_sz)
                            for f, a in zip(frames, anno)]

    frames_crop, resize_factors = zip(*crops_resize_factors)

    crop_sz = np.array([output_sz, output_sz], 'int')

    # find the bb location in the crop
    anno_crop = [transform_image_to_crop(a, a, rf, crop_sz)
                 for a, rf in zip(anno, resize_factors)]

    return frames_crop, anno_crop


def jittered_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz,
                         scale_type='original', border_type='replicate'):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """
    crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz,
                                          scale_type=scale_type, border_type=border_type)
                            for f, a in zip(frames, box_extract)]

    frames_crop, resize_factors = zip(*crops_resize_factors)

    crop_sz = np.array([output_sz, output_sz], 'int')

    # find the bb location in the crop
    box_crop = [transform_image_to_crop(a_gt, a_ex, rf, crop_sz)
                for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]

    return frames_crop, box_crop


def iou(reference, proposals):
    """Compute the IoU between a reference box with multiple proposal boxes.

    args:
        reference - Tensor of shape (1, 4).
        proposals - Tensor of shape (num_proposals, 4)

    returns:
        torch.Tensor - Tensor of shape (num_proposals,) containing IoU of reference box with each proposal box.
    """

    # Intersection box
    tl = np.maximum(reference[:, :2], proposals[:, :2])
    br = np.minimum(reference[:, :2] + reference[:, 2:], proposals[:, :2] + proposals[:, 2:])
    sz = np.clip(br - tl, 0, np.inf)

    # Area
    intersection = np.prod(sz, axis=1)
    union = np.prod(reference[:, 2:], axis=1) + np.prod(proposals[:, 2:], axis=1) - intersection

    return intersection / union


def rand_uniform(a, b, rng=None, shape=1):
    """ sample numbers uniformly between a and b.
    args:
        a - lower bound
        b - upper bound
        shape - shape of the output tensor

    returns:
        torch.Tensor - tensor of shape=shape
    """
    rand = np.random.rand if rng is None else rng.rand
    return (b - a) * rand(shape) + a


def perturb_box(box, min_iou=0.5, sigma_factor=0.1, rng=None):
    """ Perturb the input box by adding gaussian noise to the co-ordinates

     args:
        box - input box
        min_iou - minimum IoU overlap between input box and the perturbed box
        sigma_factor - amount of perturbation, relative to the box size. Can be either a single element, or a list of
                        sigma_factors, in which case one of them will be uniformly sampled. Further, each of the
                        sigma_factor element can be either a float, or a tensor
                        of shape (4,) specifying the sigma_factor per co-ordinate

    returns:
        torch.Tensor - the perturbed box
    """
    if rng is None:
        rng = np.random

    if isinstance(sigma_factor, list):
        # If list, sample one sigma_factor as current sigma factor
        c_sigma_factor = rng.choice(sigma_factor)
    else:
        c_sigma_factor = sigma_factor

    if not isinstance(c_sigma_factor, np.ndarray):
        c_sigma_factor = c_sigma_factor * np.ones(4)

    perturb_factor = np.sqrt(box[2] * box[3]) * c_sigma_factor

    # multiple tries to ensure that the perturbed box has iou > min_iou with the input box
    for i_ in range(100):
        c_x = box[0] + 0.5 * box[2]
        c_y = box[1] + 0.5 * box[3]
        c_x_per = rng.normal(c_x, perturb_factor[0])
        c_y_per = rng.normal(c_y, perturb_factor[1])

        w_per = rng.normal(box[2], perturb_factor[2])
        h_per = rng.normal(box[3], perturb_factor[3])

        if w_per <= 1:
            w_per = box[2] * rand_uniform(0.15, 0.5, rng)[0]

        if h_per <= 1:
            h_per = box[3] * rand_uniform(0.15, 0.5, rng)[0]

        box_per = np.round(np.array([c_x_per - 0.5 * w_per, c_y_per - 0.5 * h_per, w_per, h_per]))

        if box_per[2] <= 1:
            box_per[2] = box[2] * rand_uniform(0.15, 0.5, rng)

        if box_per[3] <= 1:
            box_per[3] = box[3] * rand_uniform(0.15, 0.5, rng)

        box_iou = iou(np.reshape(box, (1, 4)), np.reshape(box_per, (1, 4)))

        # if there is sufficient overlap, return
        if box_iou > min_iou:
            return box_per, box_iou

        # else reduce the perturb factor
        perturb_factor *= 0.9

    return box_per, box_iou
