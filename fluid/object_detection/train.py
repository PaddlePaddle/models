import paddle
import paddle.fluid as fluid
import reader
import image_util
import load_model as load_model
from mobilenet_ssd import mobile_net
from utility import add_arguments, print_arguments
import os
import time
import numpy as np
import argparse
import functools

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('learning_rate', float, 0.001, "Learning rate.")
add_arg('batch_size', int, 32, "Minibatch size.")
add_arg('num_passes', int, 0, "Epoch number.")
add_arg('nms_threshold', float, 0.5, "NMS threshold.")
add_arg('parallel', bool, False, "Whether use parallel training.")
add_arg('use_gpu', bool, False, "Whether use GPU.")
add_arg('data_dir', str, './data/COCO17', "Root path of data")
add_arg('train_file_list', str, 'annotations/instances_train2017.json',
        "train file list")
add_arg('val_file_list', str, 'annotations/instances_val2017.json',
        "vaild file list")
add_arg('model_save_dir', str, 'model_COCO17', "where to save model")

add_arg('dataset', str, 'coco', "coco or pascalvoc")
add_arg(
    'is_toy', int, 4,
    "Is Toy for quick debug, 0 means using all data, while n means using only n sample"
)
add_arg('label_file', str, 'label_list',
        "Lable file which lists all label name")
add_arg('apply_distort', bool, True, "Whether apply distort")
add_arg('apply_expand', bool, False, "Whether appley expand")
add_arg('resize_h', int, 300, "resize image size")
add_arg('resize_w', int, 300, "resize image size")
add_arg('mean_value_B', float, 127.5,
        "mean value which will be subtracted")  #123.68
add_arg('mean_value_G', float, 127.5,
        "mean value which will be subtracted")  #116.78
add_arg('mean_value_R', float, 127.5,
        "mean value which will be subtracted")  #103.94


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
    if data_args.dataset == 'coco':
        num_classes = 91
    elif data_args.dataset == 'pascalvoc':
        num_classes = 21

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

    if args.parallel:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places)
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
                locs, confs, box, box_var, nms_threshold=args.nms_threshold)
            loss = fluid.layers.reduce_sum(loss)
            pd.write_output(loss)
            pd.write_output(nmsed_out)

        loss, nmsed_out = pd()
        loss = fluid.layers.mean(loss)
    else:
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
            ap_version='integral')

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

    #load_model.load_and_set_vars(place)
    load_model.load_paddlev1_vars(place)
    train_reader = paddle.batch(
        reader.train(data_args, train_file_list), batch_size=batch_size)
    test_reader = paddle.batch(
        reader.test(data_args, val_file_list), batch_size=batch_size)
    if data_args.dataset == 'coco':
        feeder = fluid.DataFeeder(
            place=place, feed_list=[image, gt_box, gt_label, gt_iscrowd, gt_image_info])
    elif data_args.dataset == 'pascalvoc':
        feeder = fluid.DataFeeder(
            place=place, feed_list=[image, gt_box, gt_label, difficult])

    def test(pass_id):
        if data_args.dataset == 'coco':
            dts_res = []
            import json

            for batch_id, data in enumerate(test_reader()):
                nmsed_out_v = exe.run(fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[nmsed_out],
                                        return_numpy=False)
                lod = nmsed_out_v[0].lod()[0]
                nmsed_out_v = np.array(nmsed_out_v[0])
                real_batch_size = min(batch_size, len(data))
                assert (len(lod) == real_batch_size + 1), \
                "Error Lod Tensor offset dimension. Lod({}) vs. batch_size({})".format(len(lod), batch_size)
                k = 0
                for i in range(real_batch_size):
                    dt_num_this_img = lod[i + 1] - lod[i]
                    image_id = int(data[i][4][0])
                    image_width = int(data[i][4][1])
                    image_height = int(data[i][4][2])
                    for j in range(dt_num_this_img):
                        dt = nmsed_out_v[k]
                        k = k + 1
                        category_id, score, xmin, ymin, xmax, ymax = dt.tolist()
                        xmin = max(min(xmin, 1.0), 0.0) * image_width
                        ymin = max(min(ymin, 1.0), 0.0) * image_height
                        xmax = max(min(xmax, 1.0), 0.0) * image_width
                        ymax = max(min(ymax, 1.0), 0.0) * image_height
                        w = xmax - xmin
                        h = ymax - ymin
                        bbox = [xmin, ymin, w, h]
                        dt_res = {
                            'image_id' : image_id,
                            'category_id' : category_id,
                            'bbox' : bbox,
                            'score' : score
                        }
                        dts_res.append(dt_res)
            
            with open("detection_result.json", 'w') as outfile:
                json.dump(dts_res, outfile)
            print("start evaluate using coco api")
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            cocoGt=COCO(os.path.join(args.data_dir,args.val_file_list))
            cocoDt=cocoGt.loadRes("detection_result.json")
            cocoEval = COCOeval(cocoGt,cocoDt,"bbox")
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

        elif data_args.dataset == 'pascalvoc':
            _, accum_map = map_eval.get_map_var()
            map_eval.reset(exe)
            test_map = None
            for _, data in enumerate(test_reader()):
                test_map = exe.run(test_program,
                                   feed=feeder.feed(data),
                                   fetch_list=[accum_map])
            print("Test {0}, map {1}".format(pass_id, test_map[0]))
    test(-1)
    for pass_id in range(num_passes):
        start_time = time.time()
        prev_start_time = start_time
        end_time = 0
        for batch_id, data in enumerate(train_reader()):
            prev_start_time = start_time
            start_time = time.time()
            #print("Batch {} start at {:.2f}".format(batch_id, start_time))
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
            fluid.io.save_inference_model(model_path, ['image'], [nmsed_out],
                                          exe)


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    data_args = reader.Settings(
        dataset=args.dataset,  # coco or pascalvoc
        toy=args.is_toy,
        data_dir=args.data_dir,
        label_file=args.label_file,
        apply_distort=args.apply_distort,
        apply_expand=args.apply_expand,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        mean_value=[args.mean_value_B, args.mean_value_G, args.mean_value_R])
    train(
        args,
        train_file_list=args.train_file_list,
        val_file_list=args.val_file_list,
        data_args=data_args,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_passes=args.num_passes,
        model_save_dir=args.model_save_dir)
