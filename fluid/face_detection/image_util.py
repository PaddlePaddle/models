from PIL import Image, ImageEnhance, ImageDraw
from PIL import ImageFile
import numpy as np
import random
import math

ImageFile.LOAD_TRUNCATED_IMAGES = True  #otherwise IOError raised image file is truncated


class sampler():
    def __init__(self,
                 max_sample,
                 max_trial,
                 min_scale,
                 max_scale,
                 min_aspect_ratio,
                 max_aspect_ratio,
                 min_jaccard_overlap,
                 max_jaccard_overlap,
                 min_object_coverage,
                 max_object_coverage,
                 use_square=False):
        self.max_sample = max_sample
        self.max_trial = max_trial
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_jaccard_overlap = min_jaccard_overlap
        self.max_jaccard_overlap = max_jaccard_overlap
        self.min_object_coverage = min_object_coverage
        self.max_object_coverage = max_object_coverage
        self.use_square = use_square


class bbox():
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def intersect_bbox(bbox1, bbox2):
    if bbox2.xmin > bbox1.xmax or bbox2.xmax < bbox1.xmin or \
        bbox2.ymin > bbox1.ymax or bbox2.ymax < bbox1.ymin:
        intersection_box = bbox(0.0, 0.0, 0.0, 0.0)
    else:
        intersection_box = bbox(
            max(bbox1.xmin, bbox2.xmin),
            max(bbox1.ymin, bbox2.ymin),
            min(bbox1.xmax, bbox2.xmax), min(bbox1.ymax, bbox2.ymax))
    return intersection_box


def bbox_coverage(bbox1, bbox2):
    inter_box = intersect_bbox(bbox1, bbox2)
    intersect_size = bbox_area(inter_box)

    if intersect_size > 0:
        bbox1_size = bbox_area(bbox1)
        return intersect_size / bbox1_size
    else:
        return 0.


def bbox_area(src_bbox):
    if src_bbox.xmax < src_bbox.xmin or src_bbox.ymax < src_bbox.ymin:
        return 0.
    else:
        width = src_bbox.xmax - src_bbox.xmin
        height = src_bbox.ymax - src_bbox.ymin
        return width * height


def generate_sample(sampler, image_width, image_height):
    scale = random.uniform(sampler.min_scale, sampler.max_scale)
    aspect_ratio = random.uniform(sampler.min_aspect_ratio,
                                  sampler.max_aspect_ratio)
    aspect_ratio = max(aspect_ratio, (scale**2.0))
    aspect_ratio = min(aspect_ratio, 1 / (scale**2.0))

    bbox_width = scale * (aspect_ratio**0.5)
    bbox_height = scale / (aspect_ratio**0.5)

    # guarantee a squared image patch after cropping
    if sampler.use_square:
        if image_height < image_width:
            bbox_width = bbox_height * image_height / image_width
        else:
            bbox_height = bbox_width * image_width / image_height

    xmin_bound = 1 - bbox_width
    ymin_bound = 1 - bbox_height
    xmin = random.uniform(0, xmin_bound)
    ymin = random.uniform(0, ymin_bound)
    xmax = xmin + bbox_width
    ymax = ymin + bbox_height
    sampled_bbox = bbox(xmin, ymin, xmax, ymax)
    return sampled_bbox


def jaccard_overlap(sample_bbox, object_bbox):
    if sample_bbox.xmin >= object_bbox.xmax or \
            sample_bbox.xmax <= object_bbox.xmin or \
            sample_bbox.ymin >= object_bbox.ymax or \
            sample_bbox.ymax <= object_bbox.ymin:
        return 0
    intersect_xmin = max(sample_bbox.xmin, object_bbox.xmin)
    intersect_ymin = max(sample_bbox.ymin, object_bbox.ymin)
    intersect_xmax = min(sample_bbox.xmax, object_bbox.xmax)
    intersect_ymax = min(sample_bbox.ymax, object_bbox.ymax)
    intersect_size = (intersect_xmax - intersect_xmin) * (
        intersect_ymax - intersect_ymin)
    sample_bbox_size = bbox_area(sample_bbox)
    object_bbox_size = bbox_area(object_bbox)
    overlap = intersect_size / (
        sample_bbox_size + object_bbox_size - intersect_size)
    return overlap


def satisfy_sample_constraint(sampler, sample_bbox, bbox_labels):
    if sampler.min_jaccard_overlap == 0 and sampler.max_jaccard_overlap == 0:
        has_jaccard_overlap = False
    else:
        has_jaccard_overlap = True
    if sampler.min_object_coverage == 0 and sampler.max_object_coverage == 0:
        has_object_coverage = False
    else:
        has_object_coverage = True

    if not has_jaccard_overlap and not has_object_coverage:
        return True
    found = False
    for i in range(len(bbox_labels)):
        object_bbox = bbox(bbox_labels[i][1], bbox_labels[i][2],
                           bbox_labels[i][3], bbox_labels[i][4])
        if has_jaccard_overlap:
            overlap = jaccard_overlap(sample_bbox, object_bbox)
            if sampler.min_jaccard_overlap != 0 and \
                    overlap < sampler.min_jaccard_overlap:
                continue
            if sampler.max_jaccard_overlap != 0 and \
                    overlap > sampler.max_jaccard_overlap:
                continue
            found = True
        if has_object_coverage:
            object_coverage = bbox_coverage(object_bbox, sample_bbox)
            if sampler.min_object_coverage != 0 and \
                    object_coverage < sampler.min_object_coverage:
                continue
            if sampler.max_object_coverage != 0 and \
                    object_coverage > sampler.max_object_coverage:
                continue
            found = True
        if found:
            return True
    return found


def generate_batch_samples(batch_sampler, bbox_labels, image_width,
                           image_height):
    sampled_bbox = []
    index = []
    c = 0
    for sampler in batch_sampler:
        found = 0
        for i in range(sampler.max_trial):
            if found >= sampler.max_sample:
                break
            sample_bbox = generate_sample(sampler, image_width, image_height)
            if satisfy_sample_constraint(sampler, sample_bbox, bbox_labels):
                sampled_bbox.append(sample_bbox)
                found = found + 1
                index.append(c)
        c = c + 1
    return sampled_bbox


def clip_bbox(src_bbox):
    src_bbox.xmin = max(min(src_bbox.xmin, 1.0), 0.0)
    src_bbox.ymin = max(min(src_bbox.ymin, 1.0), 0.0)
    src_bbox.xmax = max(min(src_bbox.xmax, 1.0), 0.0)
    src_bbox.ymax = max(min(src_bbox.ymax, 1.0), 0.0)
    return src_bbox


def meet_emit_constraint(src_bbox, sample_bbox):
    center_x = (src_bbox.xmax + src_bbox.xmin) / 2
    center_y = (src_bbox.ymax + src_bbox.ymin) / 2
    if center_x >= sample_bbox.xmin and \
        center_x <= sample_bbox.xmax and \
        center_y >= sample_bbox.ymin and \
        center_y <= sample_bbox.ymax:
        return True
    return False


def project_bbox(object_bbox, sample_bbox):
    if object_bbox.xmin >= sample_bbox.xmax or \
       object_bbox.xmax <= sample_bbox.xmin or \
       object_bbox.ymin >= sample_bbox.ymax or \
       object_bbox.ymax <= sample_bbox.ymin:
        return False
    else:
        proj_bbox = bbox(0, 0, 0, 0)
        sample_width = sample_bbox.xmax - sample_bbox.xmin
        sample_height = sample_bbox.ymax - sample_bbox.ymin
        proj_bbox.xmin = (object_bbox.xmin - sample_bbox.xmin) / sample_width
        proj_bbox.ymin = (object_bbox.ymin - sample_bbox.ymin) / sample_height
        proj_bbox.xmax = (object_bbox.xmax - sample_bbox.xmin) / sample_width
        proj_bbox.ymax = (object_bbox.ymax - sample_bbox.ymin) / sample_height
        proj_bbox = clip_bbox(proj_bbox)
        if bbox_area(proj_bbox) > 0:
            return proj_bbox
        else:
            return False


def transform_labels(bbox_labels, sample_bbox):
    sample_labels = []
    for i in range(len(bbox_labels)):
        sample_label = []
        object_bbox = bbox(bbox_labels[i][1], bbox_labels[i][2],
                           bbox_labels[i][3], bbox_labels[i][4])
        if not meet_emit_constraint(object_bbox, sample_bbox):
            continue
        proj_bbox = project_bbox(object_bbox, sample_bbox)
        if proj_bbox:
            sample_label.append(bbox_labels[i][0])
            sample_label.append(float(proj_bbox.xmin))
            sample_label.append(float(proj_bbox.ymin))
            sample_label.append(float(proj_bbox.xmax))
            sample_label.append(float(proj_bbox.ymax))
            sample_label = sample_label + bbox_labels[i][5:]
            sample_labels.append(sample_label)
    return sample_labels


def crop_image(img, bbox_labels, sample_bbox, image_width, image_height):
    sample_bbox = clip_bbox(sample_bbox)
    xmin = int(sample_bbox.xmin * image_width)
    xmax = int(sample_bbox.xmax * image_width)
    ymin = int(sample_bbox.ymin * image_height)
    ymax = int(sample_bbox.ymax * image_height)
    sample_img = img[ymin:ymax, xmin:xmax]
    sample_labels = transform_labels(bbox_labels, sample_bbox)
    return sample_img, sample_labels


def random_brightness(img, settings):
    prob = random.uniform(0, 1)
    if prob < settings._brightness_prob:
        delta = random.uniform(-settings._brightness_delta,
                               settings._brightness_delta) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
    return img


def random_contrast(img, settings):
    prob = random.uniform(0, 1)
    if prob < settings._contrast_prob:
        delta = random.uniform(-settings._contrast_delta,
                               settings._contrast_delta) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
    return img


def random_saturation(img, settings):
    prob = random.uniform(0, 1)
    if prob < settings._saturation_prob:
        delta = random.uniform(-settings._saturation_delta,
                               settings._saturation_delta) + 1
        img = ImageEnhance.Color(img).enhance(delta)
    return img


def random_hue(img, settings):
    prob = random.uniform(0, 1)
    if prob < settings._hue_prob:
        delta = random.uniform(-settings._hue_delta, settings._hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
    return img


def distort_image(img, settings):
    prob = random.uniform(0, 1)
    # Apply different distort order
    if prob > 0.5:
        img = random_brightness(img, settings)
        img = random_contrast(img, settings)
        img = random_saturation(img, settings)
        img = random_hue(img, settings)
    else:
        img = random_brightness(img, settings)
        img = random_saturation(img, settings)
        img = random_hue(img, settings)
        img = random_contrast(img, settings)
    return img


def expand_image(img, bbox_labels, img_width, img_height, settings):
    prob = random.uniform(0, 1)
    if prob < settings._expand_prob:
        if settings._expand_max_ratio - 1 >= 0.01:
            expand_ratio = random.uniform(1, settings._expand_max_ratio)
            height = int(img_height * expand_ratio)
            width = int(img_width * expand_ratio)
            h_off = math.floor(random.uniform(0, height - img_height))
            w_off = math.floor(random.uniform(0, width - img_width))
            expand_bbox = bbox(-w_off / img_width, -h_off / img_height,
                               (width - w_off) / img_width,
                               (height - h_off) / img_height)
            expand_img = np.ones((height, width, 3))
            expand_img = np.uint8(expand_img * np.squeeze(settings._img_mean))
            expand_img = Image.fromarray(expand_img)
            expand_img.paste(img, (int(w_off), int(h_off)))
            bbox_labels = transform_labels(bbox_labels, expand_bbox)
            return expand_img, bbox_labels, width, height
    return img, bbox_labels, img_width, img_height
