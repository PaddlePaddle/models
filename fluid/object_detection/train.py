import os
import time
import numpy as np
import argparse
import functools
import shutil

import paddle
import paddle.fluid as fluid
import reader
from mobilenet_ssd import mobile_net
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('learning_rate',    float, 0.001,     "Learning rate.")
add_arg('batch_size',       int,   32,        "Minibatch size.")
add_arg('num_passes',       int,   120,       "Epoch number.")
add_arg('use_gpu',          bool,  True,      "Whether use GPU.")
add_arg('parallel',         bool,  True,      "Parallel.")
add_arg('use_nccl',         bool,  True,      "NCCL.")
add_arg('dataset',          str,   'pascalvoc', "coco2014, coco2017, and pascalvoc.")
add_arg('model_save_dir',   str,   'model',     "The path to save model.")
add_arg('pretrained_model', str,   'pretrained/ssd_mobilenet_v1_coco/', "The init model path.")
add_arg('apply_distort',    bool,  True,   "Whether apply distort.")
add_arg('apply_expand',     bool,  False,  "Whether appley expand.")
add_arg('nms_threshold',    float, 0.45,   "NMS threshold.")
add_arg('ap_version',       str,   'integral',   "integral, 11point.")
add_arg('resize_h',         int,   300,    "The resized image height.")
add_arg('resize_w',         int,   300,    "The resized image height.")
add_arg('mean_value_B',     float, 127.5,  "Mean value for B channel which will be subtracted.")  #123.68
add_arg('mean_value_G',     float, 127.5,  "Mean value for G channel which will be subtracted.")  #116.78
add_arg('mean_value_R',     float, 127.5,  "Mean value for R channel which will be subtracted.")  #103.94
add_arg('is_toy',           int,   0, "Toy for quick debug, 0 means using all data, while n means using only n sample.")
#yapf: enable

def parallel_do(args,
                train_file_list,
                val_file_list,
                data_args,
                learning_rate,
                batch_size,
                num_passes,
                model_save_dir,
                pretrained_model=None):
    image_shape = [3, data_args.resize_h, data_args.resize_w]
    if data_args.dataset == 'coco':
        num_classes = 81
    elif data_args.dataset == 'pascalvoc':
        num_classes = 21

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[1], dtype='int32', lod_level=1)
    difficult = fluid.layers.data(
        name='gt_difficult', shape=[1], dtype='int32', lod_level=1)

    if args.parallel:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places, use_nccl=args.use_nccl)
        with pd.do():
            image_ = pd.read_input(image)
            gt_box_ = pd.read_input(gt_box)
            gt_label_ = pd.read_input(gt_label)
            difficult_ = pd.read_input(difficult)
            locs, confs, box, box_var = mobile_net(num_classes, image_,
                                                   image_shape)
            loss = fluid.layers.ssd_loss(locs, confs, gt_box_, gt_label_, box,
                                         box_var)
            nmsed_out = fluid.layers.detection_output(
                locs, confs, box, box_var, nms_threshold=0.45)
            loss = fluid.layers.reduce_sum(loss)
            pd.write_output(loss)
            pd.write_output(nmsed_out)

        loss, nmsed_out = pd()
        loss = fluid.layers.mean(loss)
    else:
        locs, confs, box, box_var = mobile_net(num_classes, image, image_shape)
        nmsed_out = fluid.layers.detection_output(
            locs, confs, box, box_var, nms_threshold=0.45)
        loss = fluid.layers.ssd_loss(locs, confs, gt_box, gt_label, box,
                                     box_var)
        loss = fluid.layers.reduce_sum(loss)

    test_program = fluid.default_main_program().clone(for_test=True)
    with fluid.program_guard(test_program):
        map_eval = fluid.evaluator.DetectionMAP(
            nmsed_out,
            gt_label,
            gt_box,
            difficult,
            num_classes,
            overlap_threshold=0.5,
            evaluate_difficult=False,
            ap_version=args.ap_version)

    if data_args.dataset == 'coco':
        # learning rate decay in 12, 19 pass, respectively
        if '2014' in train_file_list:
            boundaries = [82783 / batch_size * 12, 82783 / batch_size * 19]
        elif '2017' in train_file_list:
            boundaries = [118287 / batch_size * 12, 118287 / batch_size * 19]
    elif data_args.dataset == 'pascalvoc':
        boundaries = [40000, 60000]
    values = [learning_rate, learning_rate * 0.5, learning_rate * 0.25]
    optimizer = fluid.optimizer.RMSProp(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        regularization=fluid.regularizer.L2Decay(0.00005), )

    optimizer.minimize(loss)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))
        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    train_reader = paddle.batch(
        reader.train(data_args, train_file_list), batch_size=batch_size)
    test_reader = paddle.batch(
        reader.test(data_args, val_file_list), batch_size=batch_size)
    feeder = fluid.DataFeeder(
        place=place, feed_list=[image, gt_box, gt_label, difficult])

    def test(pass_id):
        _, accum_map = map_eval.get_map_var()
        map_eval.reset(exe)
        test_map = None
        for data in test_reader():
            test_map = exe.run(test_program,
                               feed=feeder.feed(data),
                               fetch_list=[accum_map])
        print("Pass {0}, test map {1}".format(pass_id, test_map[0]))

    for pass_id in range(num_passes):
        start_time = time.time()
        prev_start_time = start_time
        end_time = 0
        for batch_id, data in enumerate(train_reader()):
            prev_start_time = start_time
            start_time = time.time()
            loss_v = exe.run(fluid.default_main_program(),
                             feed=feeder.feed(data),
                             fetch_list=[loss])
            end_time = time.time()
            if batch_id % 20 == 0:
                print("Pass {0}, batch {1}, loss {2}, time {3}".format(
                    pass_id, batch_id, loss_v[0], start_time - prev_start_time))
        test(pass_id)

        if pass_id % 10 == 0 or pass_id == num_passes - 1:
            model_path = os.path.join(model_save_dir, str(pass_id))
            print 'save models to %s' % (model_path)
            fluid.io.save_persistables(exe, model_path)


def parallel_exe(args,
                 train_file_list,
                 val_file_list,
                 data_args,
                 learning_rate,
                 batch_size,
                 num_passes,
                 model_save_dir,
                 pretrained_model=None):
    image_shape = [3, data_args.resize_h, data_args.resize_w]
    if 'coco' in data_args.dataset:
        num_classes = 91
    elif 'pascalvoc' in data_args.dataset:
        num_classes = 21

    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[1], dtype='int32', lod_level=1)
    difficult = fluid.layers.data(
        name='gt_difficult', shape=[1], dtype='int32', lod_level=1)
    gt_iscrowd = fluid.layers.data(
        name='gt_iscrowd', shape=[1], dtype='int32', lod_level=1)
    gt_image_info = fluid.layers.data(
        name='gt_image_id', shape=[3], dtype='int32', lod_level=1)

    locs, confs, box, box_var = mobile_net(num_classes, image, image_shape)
    nmsed_out = fluid.layers.detection_output(
        locs, confs, box, box_var, nms_threshold=args.nms_threshold)
    loss = fluid.layers.ssd_loss(locs, confs, gt_box, gt_label, box,
                                 box_var)
    loss = fluid.layers.reduce_sum(loss)

    test_program = fluid.default_main_program().clone(for_test=True)
    with fluid.program_guard(test_program):
        map_eval = fluid.evaluator.DetectionMAP(
            nmsed_out,
            gt_label,
            gt_box,
            difficult,
            num_classes,
            overlap_threshold=0.5,
            evaluate_difficult=False,
            ap_version=args.ap_version)

    if 'coco' in data_args.dataset:
        # learning rate decay in 12, 19 pass, respectively
        if '2014' in train_file_list:
            epocs = 82783 / batch_size
            boundaries = [epocs * 12, epocs * 19]
        elif '2017' in train_file_list:
            epocs = 118287 / batch_size
            boundaries = [epcos * 12, epocs * 19]
        values = [
            learning_rate, learning_rate * 0.5, learning_rate * 0.25
        ]
    elif data_args.dataset == 'pascalvoc':
        epocs = 19200 / batch_size
        boundaries = [epocs * 40, epocs * 60, epocs * 80, epocs * 100]
        values = [
            learning_rate, learning_rate * 0.5, learning_rate * 0.25,
            learning_rate * 0.1, learning_rate * 0.01
        ]
    optimizer = fluid.optimizer.RMSProp(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        regularization=fluid.regularizer.L2Decay(0.00005), )

    optimizer.minimize(loss)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))
        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    if args.parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_gpu, loss_name=loss.name)

    train_reader = paddle.batch(
        reader.train(data_args, train_file_list), batch_size=batch_size)
    test_reader = paddle.batch(
        reader.test(data_args, val_file_list), batch_size=batch_size)
    feeder = fluid.DataFeeder(
        place=place, feed_list=[image, gt_box, gt_label, difficult])

    def save_model(postfix):
        model_path = os.path.join(model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        print 'save models to %s' % (model_path)
        fluid.io.save_persistables(exe, model_path)

    best_map = 0.

    def test(pass_id, best_map):
        _, accum_map = map_eval.get_map_var()
        map_eval.reset(exe)
        for batch_id, data in enumerate(test_reader()):
            test_map = exe.run(test_program,
                               feed=feeder.feed(data),
                               fetch_list=[accum_map])
            if batch_id % 20 == 0:
                print("Batch {0}, map {1}".format(batch_id, test_map[0]))
        if test_map[0] > best_map:
            best_map = test_map[0]
            save_model('best_model')
        print("Pass {0}, test map {1}".format(pass_id, test_map[0]))
        return best_map

    for pass_id in range(num_passes):
        start_time = time.time()
        prev_start_time = start_time
        end_time = 0
        for batch_id, data in enumerate(train_reader()):
            prev_start_time = start_time
            start_time = time.time()
            if len(data) < devices_num: continue
            if args.parallel:
                loss_v, = train_exe.run(fetch_list=[loss.name],
                                        feed_dict=feeder.feed(data))
            else:
                loss_v, = exe.run(fluid.default_main_program(),
                                  feed=feeder.feed(data),
                                  fetch_list=[loss])
            end_time = time.time()
            loss_v = np.mean(np.array(loss_v))
            if batch_id % 20 == 0:
                print("Pass {0}, batch {1}, loss {2}, time {3}".format(
                    pass_id, batch_id, loss_v, start_time - prev_start_time))
        best_map = test(pass_id, best_map)
        if pass_id % 10 == 0 or pass_id == num_passes - 1:
            save_model(str(pass_id))
    print("Best test map {0}".format(best_map))


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    data_dir = 'data/pascalvoc'
    train_file_list = 'trainval.txt'
    val_file_list = 'test.txt'
    label_file = 'label_list'
    model_save_dir = args.model_save_dir
    if 'coco' in args.dataset:
        data_dir = './data/coco'
        if '2014' in args.dataset:
            train_file_list = 'annotations/instances_train2014.json'
            val_file_list = 'annotations/instances_minival2014.json'
        elif '2017' in args.dataset:
            train_file_list = 'annotations/instances_train2017.json'
            val_file_list = 'annotations/instances_val2017.json'

    data_args = reader.Settings(
        dataset=args.dataset,
        data_dir=data_dir,
        label_file=label_file,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        mean_value=[args.mean_value_B, args.mean_value_G, args.mean_value_R],
        apply_distort=args.apply_distort,
        apply_expand=args.apply_expand,
        ap_version = args.ap_version,
        toy=args.is_toy)
    method = parallel_exe
    method(
        args,
        train_file_list=train_file_list,
        val_file_list=val_file_list,
        data_args=data_args,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_passes=args.num_passes,
        model_save_dir=model_save_dir,
        pretrained_model=args.pretrained_model)
