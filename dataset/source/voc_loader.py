# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import xml.etree.ElementTree as ET

def load(anno_path, sample_num=-1):
    """ Load VOC records with annotations in 
        xml directory 'anno_path'

        Notes:
        ${anno_path}/ImageSets/Main/train.txt must contains xml file names for annotations
        ${anno_path}/Annotations/xxx.xml must contain annotation info for one record

    Args:
        @anno_path (str): root directory for voc annotation data
        @sample_num (int): number of samples to load, -1 means all

    Returns:
        (records, catname2clsid)
        'records' is list of dict whose structure is:
        {
            'im_file': im_fname, # image file name
            'im_id': im_id, # image id
            'h': im_h, # height of image
            'w': im_w, # width
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': gt_poly,
        }
        'catname2clsid' is a dict to map category name to class id

    """

    txt_file = os.path.join(anno_path, 'ImageSets',
                                  'Main', 'train.txt')
    xml_path = os.path.join(xml_path, 'Annotations')
    assert os.path.isfile(txt_file) and \
           os.path.isdir(xml_path), 'invalid xml path'

    records = []
    ct = 0

    # mapping category name to class id
    # background:0, first_class:1, second_class:2, ...
    cname2cid = {}
    with open(txt_file, 'r') as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            fname = line.strip() + '.xml'
            xml_file = os.path.join(annot_path, fname)
            if not os.path.isfile(xml_file):
                continue

            tree = ET.parse(xml_file)
            im_fname = tree.find('filename').text
            if tree.find('id') is None:
                im_id = ct
            else:
                im_id = int(tree.find('id').text)

            objs = tree.findall('object')
            im_w = float(tree.find('size').find('width').text)
            im_h = float(tree.find('size').find('height').text)
            gt_bbox = np.zeros((len(objs), 4), dtype=np.float32)
            gt_class = np.zeros((len(objs), ), dtype=np.int32)
            is_crowd = np.zeros((len(objs), ), dtype=np.int32)
            for i, obj in enumerate(objs):
                cname = obj.find('name').text
                if cname not in cname2cid:
                    cname2cid[cname] = len(cname2cid)+1
                gt_class[i] = cname2cid[cname]
                x1 = float(obj.find('bndbox').find('xmin').text)
                y1 = float(obj.find('bndbox').find('ymin').text)
                x2 = float(obj.find('bndbox').find('xmax').text)
                y2 = float(obj.find('bndbox').find('ymax').text)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(im_w - 1, x2)
                y2 = min(im_h - 1, y2)
                gt_bbox[i] = [x1, y1, x2, y2]
                is_crowd[i] = 0

            voc_rec = {
                'im_file': im_fname,
                'im_id': im_id,
                'h': im_h,
                'w': im_w,
                'is_crowd': is_crowd,
                'gt_class': gt_class,
                'gt_bbox': gt_bbox,
                'gt_poly': [],
                }
            records.append(voc_rec)

            ct += 1
            if sample_num > 0 and ct >= sample_num:
                break

    assert len(records) > 0, 'not found any voc record in %s' % (anno_path)
    return [records, cname2cid]

