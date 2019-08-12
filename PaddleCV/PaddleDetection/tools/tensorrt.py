import os
import time

import numpy as np
from PIL import Image

import paddle.fluid as fluid

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data.data_feed import create_reader

from ppdet.utils.visualizer import visualize_results
from ppdet.utils.cli import ArgsParser
import ppdet.utils.voc_eval as voc_eval
import ppdet.utils.coco_eval as coco_eval

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

eval_clses = {'COCO': coco_eval, 'VOC': voc_eval}


def create_config(model_path, mode=True):
    model_file = os.path.join(model_path, '__model__')
    params_file = os.path.join(model_path, '__params__')
    config = fluid.core.AnalysisConfig(model_file, params_file)
    config.enable_use_gpu(100, 0)
    if mode == 'trt_int8':
        config.enable_tensorrt_engine(
            1 << 30,
            1,
            min_subgraph_size=40,
            precision_mode=fluid.core.AnalysisConfig.Precision.Int8,
            use_static=False,
            use_calib_mode=True)
        logger.info('Run inference by TRT INT8.')
    elif mode == 'trt_fp32':
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=40,
            precision_mode=fluid.core.AnalysisConfig.Precision.Float32,
            use_static=False)
        logger.info('Run inference by TRT FP32.')
    elif mode == 'fluid':
        logger.info('Run inference by Fluid FP32.')
    else:
        logger.fatal('Wrong mode, only support trt_int8, trt_fp32, fluid.')
    return config


def create_tensor(np_data, dtype):
    """
    Args:
        np_data (numpy.array): numpy.array data with dtype
        dtype (string): float32, int64 or int32
    """

    dtype_map = {
        'float32': fluid.core.PaddleDType.FLOAT32,
        'int64': fluid.core.PaddleDType.INT64,
        'int32': fluid.core.PaddleDType.INT32
    }

    t = fluid.core.PaddleTensor()
    t.dtype = dtype_map[dtype]
    t.shape = np_data.shape
    buf = np_data.flatten().tolist()
    t.data = fluid.core.PaddleBuf(buf)
    return t


def create_inputs(data, architecture, use_cpp_engine=True):
    im = []
    im_size = []
    im_info = []
    im_shape = []
    for ins in data:
        im.append(ins[0])
        if architecture == 'YOLOv3':
            im_size.append(ins[1])
        elif architecture == 'SSD':
            pass
        else:
            im_info.append(ins[1])
            im_shape.append(ins[3])

    im = np.array(im)
    im_size = np.array(im_size)
    im_info = np.array(im_info)
    im_shape = np.array(im_shape)

    if use_cpp_engine:
        inputs = [create_tensor(im, 'float32')]
        if architecture == 'YOLOv3':
            inputs += [create_tensor(im_size, 'int64')]
        elif architecture == 'SSD':
            pass
        else:
            inputs += [create_tensor(im_info, 'float32')]
            inputs += [create_tensor(im_shape, 'float32')]
    else:
        inputs = [im]
        if architecture == 'YOLOv3':
            inputs += [im_size]
        elif architecture == 'SSD':
            pass
        else:
            inputs += [im_info]
            inputs += [im_shape]
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
        config = create_config(FLAGS.model_path, mode=FLAGS.mode)
        predict = fluid.core.create_paddle_predictor(config)

    for i, data in enumerate(reader()):
        if FLAGS.use_cpp_engine:
            inputs = create_inputs(data, cfg.architecture, True)
            outs = predict.run(inputs)[0]
            res = {}
            np_data = np.array(outs.data.float_data()).reshape(outs.shape)
            lengths = offset_to_lengths(outs.lod)
            res['bbox'] = (np_data, lengths)

            for k, v in zip(fields[1:], data[0][1:]):
                res[k] = (np.array(v), [[len(v)]])
            results.append(res)
        else:
            inputs = create_inputs(data, cfg.architecture, False)
            outs, = exe.run(program,
                            feed={k: v
                                  for k, v in zip(feed_names, inputs)},
                            fetch_list=fetch_targets,
                            return_numpy=False,
                            use_program_cache=True)
            res = {}
            lengths = outs.recursive_sequence_lengths()
            res['bbox'] = (np.array(outs), lengths)

            for k, v in zip(fields[1:], data[0][1:]):
                res[k] = (np.array(v), [[len(v)]])
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
                                          None, is_bbox_normalized)
                output_dir = FLAGS.output_dir
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                image_name = image_path.split('/')[-1]
                name, ext = os.path.splitext(image_name)
                save_file = os.path.join(output_dir, "{}".format(name)) + ext
                logger.info("Detection bbox results save in {}".format(
                    save_file))
                image.save(save_file, quality=95)

    if cfg.metric == 'VOC':
        eval_cls.bbox_eval(
            results, cfg.num_classes, is_bbox_normalized=is_bbox_normalized)
    else:
        eval_cls.bbox_eval(results, anno_file, 'bbox.json', with_background)


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
    config = create_config(model_path, mode=FLAGS.mode)
    predict = fluid.core.create_paddle_predictor(config)

    data = reader().next()
    inputs = create_inputs(data, cfg.architecture, True)
    print('input image shape ', inputs[0].shape)

    logger.info('warmup...')
    for i in range(10):
        outs = predict.run(inputs)

    cnt = 1000
    logger.info('run benchmark...')
    t1 = time.time()
    for i in range(cnt):
        predict.run(inputs)
    t2 = time.time()

    ms = (t2 - t1) * 1000.0 / float(cnt)

    print("Inference: {} ms per image".format(ms))

    if FLAGS.visualize:
        fields = feed.fields
        imid2path = reader.imid2path
        eval_cls = eval_clses[cfg.metric]

        anno_file = getattr(feed.dataset, 'annotation', None)
        with_background = getattr(feed, 'with_background', True)
        use_default_label = getattr(feed, 'use_default_label', False)
        clsid2catid, catid2name = eval_cls.get_category_info(
            anno_file, with_background, use_default_label)

        is_bbox_normalized = True if cfg.architecture == 'SSD' else False

        outs = outs[0]
        res = {}
        lengths = offset_to_lengths(outs.lod)
        np_data = np.array(outs.data.float_data()).reshape(outs.shape)
        res['bbox'] = (np_data, lengths)
        for k, v in zip(fields[1:], data[0][1:]):
            res[k] = (np.array(v), [[len(v)]])

        bbox_results = eval_cls.bbox2out([res], clsid2catid, is_bbox_normalized)
        im_ids = res['im_id'][0]
        for im_id in im_ids:
            image_path = imid2path[int(im_id)]
            image = Image.open(image_path).convert('RGB')
            image = visualize_results(image,
                                      int(im_id), catid2name,
                                      FLAGS.draw_threshold, bbox_results, None,
                                      is_bbox_normalized)
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
        "--calib_num", type=int, default=-1, help="The calibaration number.")
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
