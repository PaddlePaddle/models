import paddle.v2 as paddle
import paddle.fluid as fluid
import reader
import load_model as load_model
from mobilenet_ssd import mobile_net
from utility import add_arguments, print_arguments
import os
import numpy as np
import argparse
import functools

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('parallel',    bool,   True,     "Whether use parallel training.")
add_arg('use_gpu',     bool,   True,     "Whether use GPU.")
# yapf: disable


def train(args,
          train_file_list,
          val_file_list,
          data_args,
          learning_rate,
          batch_size,
          num_passes,
          model_save_dir='model',
          init_model_path=None):
    image_shape = [3, data_args.resize_h, data_args.resize_w]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[1], dtype='int32', lod_level=1)
    difficult = fluid.layers.data(
        name='gt_difficult', shape=[1], dtype='int32', lod_level=1)

    if args.parallel:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places)
        with pd.do():
            image_ = pd.read_input(image)
            gt_box_ = pd.read_input(gt_box)
            gt_label_ = pd.read_input(gt_label)
            difficult_ = pd.read_input(difficult)
            locs, confs, box, box_var = mobile_net(image_, image_shape)
            loss = fluid.layers.ssd_loss(locs, confs, gt_box_, gt_label_,
                                         box, box_var)
            pd.write_output(loss)
            pd.write_output(locs)
            pd.write_output(confs)
            pd.write_output(box)
            pd.write_output(box_var)

        loss, locs, confs, box, box_var = pd()
        loss = fluid.layers.reduce_sum(loss)
    else:
        locs, confs, box, box_var = mobile_net(image, image_shape)
        nmsed_out = fluid.layers.detection_output(
            locs, mbox_confs, box, box_var, nms_threshold=0.45)
        loss = fluid.layers.ssd_loss(locs, mbox_confs, gt_box, gt_label,
                                     box, box_var)
        loss = fluid.layers.reduce_sum(loss)

    test_program = fluid.default_main_program().clone(for_test=True)
    with fluid.program_guard(test_program):
        nmsed_out = fluid.layers.detection_output(
            locs, confs, box, box_var, nms_threshold=0.45)
        map_eval = fluid.evaluator.DetectionMAP(
            nmsed_out,
            gt_label,
            gt_box,
            difficult,
            21,
            overlap_threshold=0.5,
            evaluate_difficult=False,
            ap_version='11point')

    boundaries = [40000, 60000]
    values = [0.001, 0.0005, 0.00025]
    optimizer = fluid.optimizer.RMSProp(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        regularization=fluid.regularizer.L2Decay(0.00005), )

    optimizer.minimize(loss)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    load_model.load_and_set_vars(place)
    #load_model.load_paddlev1_vars(place)
    train_reader = paddle.batch(
        reader.train(data_args, train_file_list), batch_size=batch_size)
    test_reader = paddle.batch(
        reader.test(data_args, val_file_list), batch_size=batch_size)
    feeder = fluid.DataFeeder(
        place=place, feed_list=[image, gt_box, gt_label, difficult])

    #print 'test_program ', test_program
    def test(pass_id):
        _, accum_map = map_eval.get_map_var()
        map_eval.reset(exe)
        test_map = None
        for _, data in enumerate(test_reader()):
            test_map = exe.run(test_program,
                               feed=feeder.feed(data),
                               fetch_list=[accum_map])
        print("Test {0}, map {1}".format(pass_id, test_map[0]))

    #print 'main_program ', fluid.default_main_program()
    for pass_id in range(num_passes):
        for batch_id, data in enumerate(train_reader()):
            loss_v = exe.run(fluid.default_main_program(),
                             feed=feeder.feed(data),
                             fetch_list=[loss])
            if batch_id % 20 == 0:
                print("Pass {0}, batch {1}, loss {2}"
                      .format(pass_id, batch_id, loss_v[0]))
        test(pass_id)

        if pass_id % 10 == 0:
            model_path = os.path.join(model_save_dir, str(pass_id))
            print 'save models to %s' % (model_path)
            fluid.io.save_inference_model(model_path, ['image'], [nmsed_out],
                                          exe)


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    data_args = reader.Settings(
        data_dir='./data',
        label_file='label_list',
        apply_distort=True,
        apply_expand=True,
        resize_h=300,
        resize_w=300,
        mean_value=[127.5, 127.5, 127.5])
