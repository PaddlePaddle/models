import sys
import os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
if path not in sys.path:
    sys.path.insert(0, path)


prefix = os.path.dirname(os.path.abspath(__file__))

#coco data for testing
if sys.version.startswith('2'):
    version = 'python2'
else:
    version = 'python3'

data_root = os.path.join(prefix, 'data/coco.test.%s' % (version))

# coco data for testing
coco_data = {
    'TRAIN': {
        'ANNO_FILE': os.path.join(data_root, 'train2017.roidb'),
        'IMAGE_DIR': os.path.join(data_root, 'train2017')
    },
    'VAL': {
        'ANNO_FILE': os.path.join(data_root, 'val2017.roidb'),
        'IMAGE_DIR': os.path.join(data_root, 'val2017')
    }
}

script = os.path.join(os.path.dirname(__file__), 'data/prepare_data.sh')

if not os.path.exists(data_root):
    ret = os.system('bash %s %s' % (script, version))
    if ret != 0:
        print('not found file[%s], you should manually prepare '
            'your data using "data/prepare_data.sh"' % (anno_file))
        sys.exit(1)

