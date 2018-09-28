from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

__C.TRAIN = edict()

__C.IMG_WIDTH = 300
__C.IMG_HEIGHT = 300
__C.IMG_CHANNEL = 3
__C.CLASS_NUM = 21
__C.BACKGROUND_ID = 0

# training settings
__C.TRAIN.LEARNING_RATE = 0.001 / 4
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.NUM_PASS = 200
__C.TRAIN.L2REGULARIZATION = 0.0005 * 4
__C.TRAIN.LEARNING_RATE_DECAY_A = 0.1
__C.TRAIN.LEARNING_RATE_DECAY_B = 16551 * 80
__C.TRAIN.LEARNING_RATE_SCHEDULE = 'discexp'

__C.NET = edict()

# configuration for multibox_loss_layer
__C.NET.MBLOSS = edict()
__C.NET.MBLOSS.OVERLAP_THRESHOLD = 0.5
__C.NET.MBLOSS.NEG_POS_RATIO = 3.0
__C.NET.MBLOSS.NEG_OVERLAP = 0.5

# configuration for detection_map
__C.NET.DETMAP = edict()
__C.NET.DETMAP.OVERLAP_THRESHOLD = 0.5
__C.NET.DETMAP.EVAL_DIFFICULT = False
__C.NET.DETMAP.AP_TYPE = "11point"

# configuration for detection_output_layer
__C.NET.DETOUT = edict()
__C.NET.DETOUT.CONFIDENCE_THRESHOLD = 0.01
__C.NET.DETOUT.NMS_THRESHOLD = 0.45
__C.NET.DETOUT.NMS_TOP_K = 400
__C.NET.DETOUT.KEEP_TOP_K = 200

# configuration for priorbox_layer from conv4_3
__C.NET.CONV4 = edict()
__C.NET.CONV4.PB = edict()
__C.NET.CONV4.PB.MIN_SIZE = [30]
__C.NET.CONV4.PB.MAX_SIZE = []
__C.NET.CONV4.PB.ASPECT_RATIO = [2.]
__C.NET.CONV4.PB.VARIANCE = [0.1, 0.1, 0.2, 0.2]

# configuration for priorbox_layer from fc7
__C.NET.FC7 = edict()
__C.NET.FC7.PB = edict()
__C.NET.FC7.PB.MIN_SIZE = [60]
__C.NET.FC7.PB.MAX_SIZE = [114]
__C.NET.FC7.PB.ASPECT_RATIO = [2., 3.]
__C.NET.FC7.PB.VARIANCE = [0.1, 0.1, 0.2, 0.2]

# configuration for priorbox_layer from conv6_2
__C.NET.CONV6 = edict()
__C.NET.CONV6.PB = edict()
__C.NET.CONV6.PB.MIN_SIZE = [114]
__C.NET.CONV6.PB.MAX_SIZE = [168]
__C.NET.CONV6.PB.ASPECT_RATIO = [2., 3.]
__C.NET.CONV6.PB.VARIANCE = [0.1, 0.1, 0.2, 0.2]

# configuration for priorbox_layer from conv7_2
__C.NET.CONV7 = edict()
__C.NET.CONV7.PB = edict()
__C.NET.CONV7.PB.MIN_SIZE = [168]
__C.NET.CONV7.PB.MAX_SIZE = [222]
__C.NET.CONV7.PB.ASPECT_RATIO = [2., 3.]
__C.NET.CONV7.PB.VARIANCE = [0.1, 0.1, 0.2, 0.2]

# configuration for priorbox_layer from conv8_2
__C.NET.CONV8 = edict()
__C.NET.CONV8.PB = edict()
__C.NET.CONV8.PB.MIN_SIZE = [222]
__C.NET.CONV8.PB.MAX_SIZE = [276]
__C.NET.CONV8.PB.ASPECT_RATIO = [2., 3.]
__C.NET.CONV8.PB.VARIANCE = [0.1, 0.1, 0.2, 0.2]

# configuration for priorbox_layer from pool6
__C.NET.POOL6 = edict()
__C.NET.POOL6.PB = edict()
__C.NET.POOL6.PB.MIN_SIZE = [276]
__C.NET.POOL6.PB.MAX_SIZE = [330]
__C.NET.POOL6.PB.ASPECT_RATIO = [2., 3.]
__C.NET.POOL6.PB.VARIANCE = [0.1, 0.1, 0.2, 0.2]
