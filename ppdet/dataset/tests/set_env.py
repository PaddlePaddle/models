import sys
import os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
if path not in sys.path:
    sys.path.insert(0, path)


prefix = os.path.dirname(os.path.abspath(__file__))

#coco data for testing
coco_data = {
    'ANNO_FILE': os.path.join(prefix, \
        'data/coco.test/val2017.roidb'),
    'IMAGE_DIR': os.path.join(prefix, \
        'data/coco.test/val2017')
    }

anno_file = coco_data['ANNO_FILE']
if not os.path.exists(anno_file):
    print('not found file[%s], you should prepare '
        'your data using "data/prepare_data.sh"' % (anno_file))
    sys.exit(1)
