import os
import time

import numpy as np
from PIL import Image

import paddle.fluid as fluid

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data.data_feed import create_reader

from ppdet.utils.visualizer import visualize_results
from ppdet.utils.eval_utils import eval_results
from ppdet.utils.cli import ArgsParser
from ppdet.utils.eval_utils import eval_results
import ppdet.utils.voc_eval as voc_eval
import ppdet.utils.coco_eval as coco_eval

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

eval_clses = {'COCO': coco_eval, 'VOC': voc_eval}


def create_config(model_path, mode='fluid', batch_size=1, min_subgraph_size=3):
    model_file = os.path.join(model_path, '__model__')
    params_file = os.path.join(model_path, '__params__')
    config = fluid.core.AnalysisConfig(model_file, params_file)
    config.enable_use_gpu(100, 0)
    #config.enable_profile()
    logger.info('min_subgraph_size = %d.' % (min_subgraph_size))
    if mode == 'trt_int8':
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=fluid.core.AnalysisConfig.Precision.Int8,
            use_static=False,
            use_calib_mode=True)
        logger.info('Run inference by TRT INT8.')
    elif mode == 'trt_fp32':
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=fluid.core.AnalysisConfig.Precision.Float32,
            use_static=False)
        logger.info('Run inference by TRT FP32.')
    elif mode == 'trt_fp16':
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=fluid.core.AnalysisConfig.Precision.Half,
            use_static=False)
        logger.info('Run inference by TRT FP16.')
    elif mode == 'fluid':
        logger.info('Run inference by Fluid FP32.')
    else:
        logger.fatal('Wrong mode, only support trt_int8, trt_fp32, fluid.')
    return config


def create_tensor(np_data, dtype, use_cpp_engine=True):
    """
    Args:
        np_data (numpy.array): numpy.array data with dtype
        dtype (string): float32, int64 or int32
    """
    if not use_cpp_engine:
        return np_data

    dtype_map = {
        'float32': fluid.core.PaddleDType.FLOAT32,
        'int64': fluid.core.PaddleDType.INT64,
        'int32': fluid.core.PaddleDType.INT32
    }

    t = fluid.core.PaddleTensor()
    t.dtype = dtype_map[dtype]
    t.shape = np_data.shape
    #buf = np_data.flatten().tolist()
    t.data = fluid.core.PaddleBuf(np_data)
    return t


def batch_row2col(data, fields):
    anno_bbox = ['gt_box', 'gt_label', 'is_crowd', 'is_difficult']
    #outs = []
    #if len(data) == 1:
    #    data = data[0]
    #    im = data[0]
    #    im = im.reshape((1,) + im.shape)
    #    outs.append(im)
    #    for i in xrange(1, len(data)):
    #        outs.append(data[i].reshape(1,-1))
    outs = [[[], []] for i in range(len(data[0]))]
    for batch in data:
        for i, slot in enumerate(batch):
            if fields[i] in anno_bbox:
                outs[i][0].extend(list(slot))
                outs[i][1].append(len(slot))
            else:
                outs[i][0].append(slot)

    for i, slot in enumerate(outs):
        outs[i][0] = np.array(slot[0])
        outs[i][1] = [slot[1]]
    return outs


def create_inputs(data, architecture, use_cpp_engine=True):
    im = data['image'][0]
    inputs = [create_tensor(im, 'float32', use_cpp_engine)]

    if architecture == 'YOLOv3':
        im_size = data['im_size'][0]
        inputs += [create_tensor(im_size, 'int64', use_cpp_engine)]
    elif architecture == 'SSD':
        pass
    elif architecture == 'RetinaNet' or architecture == 'CascadeRCNN':
        im_info = data['im_info'][0]
        inputs += [create_tensor(im_info, 'float32', use_cpp_engine)]
    else:
        im_info = data['im_info'][0]
        im_shape = data['im_shape'][0]
        inputs += [create_tensor(im_info, 'float32', use_cpp_engine)]
        inputs += [create_tensor(im_shape, 'float32', use_cpp_engine)]

    return inputs


def offset_to_lengths(lod):
    offset = lod[0]
    lengths = [offset[i + 1] - offset[i] for i in range(len(offset) - 1)]
    return [lengths]


def evaluate():
    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)

    feed = create(cfg.eval_feed)
    reader = create_reader(feed)
    fields = feed.fields

    imid2path = reader.imid2path
    anno_file = getattr(feed.dataset, 'annotation', None)
    with_background = getattr(feed, 'with_background', True)
    use_default_label = getattr(feed, 'use_default_label', False)

    eval_cls = eval_clses[cfg.metric]
    clsid2catid, catid2name = eval_cls.get_category_info(
        anno_file, with_background, use_default_label)
    is_bbox_normalized = True if cfg.architecture == 'SSD' else False

    place = fluid.CPUPlace() if FLAGS.use_cpp_engine else fluid.CUDAPlace(0)

    results = []
    if not FLAGS.use_cpp_engine:
        exe = fluid.Executor(place)
        [program, feed_names, fetch_targets] = fluid.io.load_inference_model(
            dirname=FLAGS.model_path,
            executor=exe,
            model_filename='__model__',
            params_filename='__params__')
    else:
        config = create_config(
            FLAGS.model_path,
            mode=FLAGS.mode,
            batch_size=feed.batch_size,
            min_subgraph_size=FLAGS.min_subgraph_size)
        predict = fluid.core.create_paddle_predictor(config)

    t1 = time.time()
    for i, data in enumerate(reader()):
        if FLAGS.use_cpp_engine:
            data = batch_row2col(data, fields)
            data_dict = {k: v for k, v in zip(fields, data)}
            inputs = create_inputs(data_dict, cfg.architecture, True)
            outs = predict.run(inputs)[0]

            res = {}
            np_data = np.array(outs.data.float_data()).reshape(outs.shape)
            lengths = offset_to_lengths(outs.lod)
            res['bbox'] = (np_data, lengths)
            res.update(data_dict)

            results.append(res)
        else:
            data = batch_row2col(data, fields)
            data_dict = {k: v for k, v in zip(fields, data)}
            inputs = create_inputs(data_dict, cfg.architecture, False)

            outs, = exe.run(program,
                            feed={k: v
                                  for k, v in zip(feed_names, inputs)},
                            fetch_list=fetch_targets,
                            return_numpy=False,
                            use_program_cache=True)
            res = {}
            lengths = outs.recursive_sequence_lengths()
            res['bbox'] = (np.array(outs), lengths)
            res.update(data_dict)

            results.append(res)

        if i % 500 == 0:
            print('Test iter: {}.'.format(i))

        if FLAGS.visualize:
            bbox_results = eval_cls.bbox2out([res], clsid2catid,
                                             is_bbox_normalized)
            im_ids = res['im_id'][0]
            for im_id in im_ids:
                image_path = imid2path[int(im_id)]
                image = Image.open(image_path).convert('RGB')
                image = visualize_results(image,
                                          int(im_id), catid2name,
                                          FLAGS.draw_threshold, bbox_results,
                                          None)
                output_dir = FLAGS.output_dir
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                image_name = image_path.split('/')[-1]
                name, ext = os.path.splitext(image_name)
                save_file = os.path.join(output_dir, "{}".format(name)) + ext
                logger.info("Detection bbox results save in {}".format(
                    save_file))
                image.save(save_file, quality=95)

    t2 = time.time()
    ins_num = 5000.0 if cfg.metric == 'COCO' else 4952.0
    speed = 5000.0 / (t2 - t1)
    print("Inference time: {} fps".format(speed))

    resolution = None
    if 'mask' in results[0]:
        resolution = model.mask_head.resolution
    eval_results(results, feed, cfg.metric, cfg.num_classes, resolution,
                 is_bbox_normalized)


def benchmark():
    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)

    if FLAGS.infer_img is not None:
        feed = create(cfg.test_feed)
        images = [FLAGS.infer_img]
        feed.dataset.add_images(images)
    else:
        feed = create(cfg.eval_feed)

    print(cfg)

    reader = create_reader(feed)

    model_path = FLAGS.model_path
    config = create_config(
        model_path,
        mode=FLAGS.mode,
        batch_size=feed.batch_size,
        min_subgraph_size=FLAGS.min_subgraph_size)
    predict = fluid.core.create_paddle_predictor(config)

    data = reader().next()
    fields = feed.fields
    data = batch_row2col(data, fields)
    data_dict = {k: v for k, v in zip(fields, data)}

    inputs = create_inputs(data_dict, cfg.architecture, True)
    print('input image shape ', inputs[0].shape)

    logger.info('warmup...')
    for i in range(10):
        outs = predict.run(inputs)

    cnt = 100
    logger.info('run benchmark...')
    #fluid.profiler.start_profiler('GPU')
    t1 = time.time()
    for i in range(cnt):
        outs = predict.run(inputs)
    t2 = time.time()
    #fluid.profiler.stop_profiler('total', 'logs')

    ms = (t2 - t1) * 1000.0 / float(cnt)

    print("Inference: {} ms per image".format(ms))

    if FLAGS.visualize:
        imid2path = reader.imid2path
        eval_cls = eval_clses[cfg.metric]

        anno_file = getattr(feed.dataset, 'annotation', None)
        with_background = getattr(feed, 'with_background', True)
        use_default_label = getattr(feed, 'use_default_label', False)
        clsid2catid, catid2name = eval_cls.get_category_info(
            anno_file, with_background, use_default_label)

        is_bbox_normalized = True if cfg.architecture == 'SSD' else False

        outs = outs[-1]
        res = {}
        lengths = offset_to_lengths(outs.lod)
        np_data = np.array(outs.data.float_data()).reshape(outs.shape)
        res['bbox'] = (np_data, lengths)
        res.update(data_dict)

        bbox_results = eval_cls.bbox2out([res], clsid2catid, is_bbox_normalized)
        im_ids = res['im_id'][0]
        for im_id in im_ids:
            image_path = imid2path[int(im_id)]
            image = Image.open(image_path).convert('RGB')
            image = visualize_results(image,
                                      int(im_id), catid2name,
                                      FLAGS.draw_threshold, bbox_results, None)
            output_dir = FLAGS.output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            image_name = image_path.split('/')[-1]
            name, ext = os.path.splitext(image_name)
            save_file = os.path.join(output_dir, "{}".format(name)) + ext
            logger.info("Detection bbox results save in {}".format(save_file))
            image.save(save_file, quality=95)


if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "--model_path", type=str, default=None, help="model path.")
    parser.add_argument(
        "--visualize",
        action='store_true',
        default=False,
        help="Whether to visualize detection output")
    parser.add_argument(
        "--mode",
        type=str,
        default='fluid',
        help="mode can be trt_fp32, trt_int8, fluid.")
    parser.add_argument(
        "--is_eval",
        action='store_true',
        default=False,
        help="Whether to do evalution")
    parser.add_argument(
        "--use_cpp_engine",
        action='store_false',
        default=True,
        help="Whether to do evalution")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--min_subgraph_size",
        type=int,
        default=3,
        help="min_subgraph_size for TensorRT.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--infer_img", type=str, default=None, help="Image path")
    FLAGS = parser.parse_args()
    if FLAGS.is_eval:
        evaluate()
    else:
        benchmark()
