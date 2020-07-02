import os
from .base_dataset import BaseDataset
from ltr.data.image_loader import default_image_loader
import numpy as np
import json
import cv2
from collections import OrderedDict
from ltr.admin.environment import env_settings


def get_target_to_image_ratio(seq):
    anno = np.array(seq['anno'])
    img_sz = np.array(seq['image_size'])
    return np.sqrt(anno[0, 2:4].prod() / (img_sz.prod()))


class Instance(object):
    instID     = 0
    pixelCount = 0

    def __init__(self, imgNp, instID):
        if (instID ==0 ):
            return
        self.instID     = int(instID)
        self.pixelCount = int(self.getInstancePixels(imgNp, instID))

    def getInstancePixels(self, imgNp, instLabel):
        return (imgNp == instLabel).sum()

    def toDict(self):
        buildDict = {}
        buildDict["instID"]     = self.instID
        buildDict["pixelCount"] = self.pixelCount
        return buildDict

    def __str__(self):
        return "("+str(self.instID)+")"


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


def polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(min(p[::2]) for p in poly)
        x1 = max(max(p[::2]) for p in poly)
        y0 = min(min(p[1::2]) for p in poly)
        y1 = max(max(p[1::2]) for p in poly)
        boxes_from_polys[i, :] = [x0, y0, x1, y1]
    return boxes_from_polys


class YoutubeVOS(BaseDataset):
    """ Youtube-VOS dataset.

    Publication:
        
        https://arxiv.org/pdf/

    Download the dataset from https://youtube-vos.org/dataset/download
    """

    def __init__(self, root=None, filter=None, image_loader=default_image_loader, min_length=1, max_target_area=1):
        """
        args:
            root - path to the youtube-vos dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            min_length - Minimum allowed sequence length.
            max_target_area - max allowed ratio between target area and image area. Can be used to filter out targets
                                which cover complete image.
        """
        root = env_settings().youtubevos_dir if root is None else root
        super().__init__(root, image_loader)

        cache_file = os.path.join(root, 'cache.json')
        if os.path.isfile(cache_file):
            # If available, load the pre-processed cache file containing meta-info for each sequence
            with open(cache_file, 'r') as f:
                sequence_list_dict = json.load(f)

            self.sequence_list = sequence_list_dict
        else:
            # Else process the youtube-vos annotations and generate the cache file
            print('processing the youtube-vos annotations...')
            self.sequence_list = self._process_anno(root)

            with open(cache_file, 'w') as f:
                json.dump(self.sequence_list, f)
            print('cache file generated!')

        # Filter the sequences based on min_length and max_target_area in the first frame
        self.sequence_list = [x for x in self.sequence_list if len(x['anno']) >= min_length and
                              get_target_to_image_ratio(x) < max_target_area]
        self.filter = filter

    def get_name(self):
        return 'youtubevos'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        anno = np.array(self.sequence_list[seq_id]['anno'])
        target_visible = (anno[:, 2] > 0) & (anno[:, 3] > 0)
        if self.filter is not None:
            target_large = (anno[:, 2] * anno[:, 3] > 30 * 30)
            ratio = anno[:, 2] / anno[:, 3]
            target_reasonable_ratio = (10 > ratio) & (ratio > 0.1)
            target_visible = target_visible & target_reasonable_ratio & target_large
        return anno, target_visible

    def _get_frame(self, sequence, frame_id):
        vid_name = sequence['video']
        frame_number = sequence['frames'][frame_id]

        frame_path = os.path.join(self.root, 'train', 'JPEGImages', vid_name,
                                  '{:05d}.jpg'.format(frame_number))
        return self.image_loader(frame_path)

    def _get_mask(self, sequence, frame_id):
        vid_name = sequence['video']
        frame_number = sequence['frames'][frame_id]
        id = sequence['id']

        mask_path = os.path.join(self.root, 'train', 'Annotations', vid_name,
                                  '{:05d}.png'.format(frame_number))
        mask = cv2.imread(mask_path, 0)
        mask = (mask == id).astype(np.float32)
        mask = np.expand_dims(mask, axis=2)
        return mask

    def get_frames(self, seq_id, frame_ids, anno=None):
        sequence = self.sequence_list[seq_id]

        frame_list = [self._get_frame(sequence, f) for f in frame_ids]

        if anno is None:
            anno = sequence['anno']

        # Return as list of tensors
        anno_frames = [anno[f_id, :] for f_id in frame_ids]

        # added the class info to the meta info
        object_meta = OrderedDict({'object_class': sequence['class_name'],
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

    def get_frames_mask(self, seq_id, frame_ids, anno=None):
        sequence = self.sequence_list[seq_id]

        frame_list = [self._get_frame(sequence, f) for f in frame_ids]
        mask_list = [self._get_mask(sequence, f) for f in frame_ids]

        if anno is None:
            anno = sequence['anno']

        # Return as list of tensors
        anno_frames = [anno[f_id, :] for f_id in frame_ids]

        # added the class info to the meta info
        object_meta = OrderedDict({'object_class': sequence['class_name'],
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, mask_list, object_meta

    def _process_anno(self, root):
        # Builds individual tracklets
        base_anno_path = os.path.join(root, 'train', 'Annotations')

        num_obj = 0
        num_ann = 0
        all_sequences = []
        meta = json.load(open(os.path.join(base_anno_path, '../meta.json')))
        for vid_id, video in enumerate(meta['videos']):
            v = meta['videos'][video]
            frames = []
            objects = dict()
            for obj in v['objects']:
                o = v['objects'][obj]
                frames.extend(o['frames'])
            frames = sorted(set(frames))

            for frame in frames:
                file_name = os.path.join(video, frame)
                img = cv2.imread(os.path.join(base_anno_path, file_name+'.png'), 0)
                h, w = img.shape[:2]
                image_size = [w, h]

                for instanceId in np.unique(img):
                    if instanceId == 0:
                        continue
                    instanceObj = Instance(img, instanceId)
                    instanceObj_dict = instanceObj.toDict()
                    mask = (img == instanceId).astype(np.uint8)
                    if cv2.__version__[0] == '3':
                        _, contour, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    else:
                        contour, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    polygons = [c.reshape(-1).tolist() for c in contour]
                    instanceObj_dict['contours'] = [p for p in polygons if len(p) > 4]
                    if len(instanceObj_dict['contours']) and instanceObj_dict['pixelCount'] > 1000:
                        len_p = [len(p) for p in instanceObj_dict['contours']]
                        if min(len_p) <= 4:
                            print('Warning: invalid contours.')
                            continue  # skip non-instance categories

                        bbox = xyxy_to_xywh(
                            polys_to_boxes([instanceObj_dict['contours']])).tolist()[0]
                        if instanceId not in objects:
                            objects[instanceId] = \
                                {'anno': [], 'frames': [], 'image_size': image_size}
                        objects[instanceId]['anno'].append(bbox)
                        objects[instanceId]['frames'].append(int(frame))

            for obj in objects:
                new_sequence = {'video': video, 'id': int(obj), 'class_name': None,
                                'frames': objects[obj]['frames'], 'anno': objects[obj]['anno'],
                                'image_size': image_size}
                all_sequences.append(new_sequence)
        print('Youtube-VOS: ', len(all_sequences))
        return all_sequences
