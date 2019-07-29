import os
import time

import numpy as np
from PIL import Image

import paddle.fluid as fluid

from ppdet.utils.cli import ArgsParser
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data.data_feed import create_reader
import ppdet.utils.voc_eval as voc_eval
import ppdet.utils.coco_eval as coco_eval

from ppdet.utils.visualizer import visualize_results

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

eval_clses = {'COCO': coco_eval, 'VOC': voc_eval}


def get_config(model_path, mode=True):
    model_file = os.path.join(model_path, '__model__')
    params_file = os.path.join(model_path, '__params__')

    config = fluid.core.AnalysisConfig(model_file, params_file)
    config.enable_use_gpu(100, 0)

    if mode == 'trt_int8':
        config.enable_tensorrt_engine(
            1 << 30,
            1,
            precision_mode=fluid.core.AnalysisConfig.Precision.Int8,
            use_static=True,
            use_calib_mode=True)
        logger.info('Run inference by TRT INT8.')
    elif mode == 'trt_fp32':
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            precision_mode=fluid.core.AnalysisConfig.Precision.Float32,
            use_static=True)
        logger.info('Run inference by TRT FP32.')
    elif mode == 'fluid':
        logger.info('Run inference by Fluid FP32.')
    else:
        logger.fatal('Wrong mode, only support trt_int8, trt_fp32, fluid.')

    return config


def offset_to_lengths(lod):
    offset = lod[0]
    lengths = [offset[i + 1] - offset[i] for i in range(len(offset) - 1)]
    return [lengths]


def eval():
    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)

    feed = create(cfg.eval_feed)
    reader = create_reader(feed)

    fields = feed.fields

    model_path = FLAGS.model_path

    results = []
    imid2path = reader.imid2path

    eval_cls = eval_clses[cfg.metric]

    anno_file = getattr(feed.dataset, 'annotation', None)
    with_background = getattr(feed, 'with_background', True)
    use_default_label = getattr(feed, 'use_default_label', False)
    clsid2catid, catid2name = eval_cls.get_category_info(
        anno_file, with_background, use_default_label)
    is_bbox_normalized = True if cfg.architecture == 'SSD' else False

    if not FLAGS.use_cpp_engine:
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        [program, feed_names, fetch_targets] = (fluid.io.load_inference_model(
            dirname=model_path,
            executor=exe,
            model_filename='__model__',
            params_filename='__params__'))

        print(program)
        print(feed_names)
    else:
        config = get_config(model_path, mode=FLAGS.mode)
        predict = fluid.core.create_paddle_predictor(config)

    print(fields)
    for i, data in enumerate(reader()):
        if FLAGS.use_cpp_engine:
            dt = data[0][0]
            in_t = fluid.core.PaddleTensor()
            in_t.dtype = fluid.core.PaddleDType.FLOAT32
            in_t.shape = (len(data), ) + dt.shape
            buf = dt.flatten().tolist()
            in_t.data = fluid.core.PaddleBuf(buf)
            in_ts = [in_t]

            if cfg.architecture == 'YOLOv3':
                dt2 = data[0][1].astype('int64')
                in_t2 = fluid.core.PaddleTensor()
                #in_t2.dtype = fluid.core.PaddleDType.INT32
                in_t2.dtype = fluid.core.PaddleDType.INT64
                in_t2.shape = (len(data), ) + dt2.shape
                buf = dt2.flatten().tolist()
                in_t2.data = fluid.core.PaddleBuf(buf)
                in_ts += [in_t2]
                if i == 0:
                    print(in_t.shape, in_t2.shape)

            #np.save('im.npy', data[0][0])
            #np.save('im_shape.npy', data[0][1])
            #return

            outs = predict.run(in_ts)[0]

            res = {}
            lengths = offset_to_lengths(outs.lod)
            res['bbox'] = (np.array(outs.data.float_data()).reshape(outs.shape),
                           lengths)

            for k, v in zip(fields[1:], data[0][1:]):
                res[k] = (np.array(v), [[len(v)]])
            results.append(res)
        else:
            dt = data[0][0]
            in1 = dt.reshape((len(data), ) + dt.shape)
            inputs = [in1]
            if cfg.architecture == 'YOLOv3':
                dt2 = data[0][1]
                in2 = dt2.reshape((len(data), ) + dt2.shape).astype('int32')
                inputs += [in2]

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
                #logger.info("Detection bbox results save in {}".format(save_file))
                image.save(save_file, quality=95)

    if cfg.metric == 'VOC':
        eval_cls.bbox_eval(
            results, cfg.num_classes, is_bbox_normalized=is_bbox_normalized)
    else:
        eval_cls.bbox_eval(results, anno_file, 'bbox.json', with_background)


def bench():
    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)

    feed = create(cfg.eval_feed)
    reader = create_reader(feed)

    model_path = FLAGS.model_path
    config = get_config(model_path, mode=FLAGS.mode)
    predict = fluid.core.create_paddle_predictor(config)

    data = reader().next()

    dt = data[0][0]
    in_t = fluid.core.PaddleTensor()
    in_t.shape = (len(data), ) + dt.shape
    buf = dt.flatten().tolist()
    in_t.data = fluid.core.PaddleBuf(buf)

    in_ts = [in_t]
    if cfg.architecture == 'YOLOv3':
        dt2 = data[0][1].astype('int64')
        in_t2 = fluid.core.PaddleTensor()
        in_t2.dtype = fluid.core.PaddleDType.INT64
        in_t2.shape = (len(data), ) + dt2.shape
        buf = dt2.flatten().tolist()
        in_t2.data = fluid.core.PaddleBuf(buf)
        in_ts += [in_t2]

    logger.info('warmup...')
    for i in range(10):
        outs = predict.run(in_ts)

    cnt = 1000
    logger.info('run benchmark...')
    t1 = time.time()
    for i in range(cnt):
        predict.run(in_ts)
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
        res['bbox'] = (np.array(outs.data.float_data()).reshape(outs.shape),
                       lengths)
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
    FLAGS = parser.parse_args()
    if FLAGS.is_eval:
        eval()
    else:
        bench()
