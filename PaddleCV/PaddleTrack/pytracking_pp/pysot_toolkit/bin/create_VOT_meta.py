import os
import os.path as osp
import json

dataset_name = 'VOT2015'
sequence_dir = 'data/' + dataset_name

with open(osp.join(sequence_dir, 'list.txt'), 'r') as f:
  seqs = [line.strip() for line in f.readlines()]
assert len(seqs) == 60

meta = {}
for seq in seqs:
  seq_meta = {'video_dir': seq}

  if dataset_name == 'VOT2015':
    img_names = [osp.join(seq, v) for v in sorted(os.listdir(osp.join(sequence_dir, seq))) if 'jpg' in v]
  else:
    img_names = [osp.join(seq, 'color', v) for v in sorted(os.listdir(osp.join(sequence_dir, seq, 'color')))]
  seq_meta['img_names'] = img_names

  with open(osp.join(sequence_dir, seq, 'groundtruth.txt'), 'r') as f:
    gt_rect = [[float(v) for v in line.strip().split(',')] for line in f.readlines()]

  assert len(gt_rect) == len(img_names), 'num of annotations not equal to num of images'

  seq_meta['init_rect'] = gt_rect[0]
  seq_meta['gt_rect'] = gt_rect
  seq_meta['camera_motion'] = [0] * len(gt_rect)
  seq_meta['illum_change'] = [0] * len(gt_rect)
  seq_meta['motion_change'] = [0] * len(gt_rect)
  seq_meta['size_change'] = [0] * len(gt_rect)
  seq_meta['occlusion'] = [0] * len(gt_rect)
  meta[seq] = seq_meta

with open(osp.join(sequence_dir, '{}.json'.format(dataset_name)), 'w') as f:
  json.dump(meta, f)
