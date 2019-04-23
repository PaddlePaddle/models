import numpy as np
import argparse
import time

import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import paddle

import reader


def parse_args():
    parser = argparse.ArgumentParser("Run inference.")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=6,
        help='The size of a batch. (default: %(default)d)')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type. (default: %(default)s)')
    parser.add_argument(
        '--model_path',
        type=str,
        default='output/params_pass_0',
        help='A path to the model. (default: %(default)s)')
    parser.add_argument(
        '--test_data_dir',
        type=str,
        default='data/test_files',
        help='A directory with test data files. (default: %(default)s)')
    parser.add_argument(
        '--test_label_file',
        type=str,
        default='data/label_dict',
        help='A file with test labels. (default: %(default)s)')
    parser.add_argument(
        '--num_passes', type=int, default=1, help='The number of passes.')
    parser.add_argument(
        '--skip_pass_num',
        type=int,
        default=0,
        help='The first num of passes to skip in statistics calculations.')
    parser.add_argument(
        '--profile', action='store_true', help='If set, do profiling.')
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def load_reverse_dict(dict_path):
    return dict((idx, line.strip().split("\t")[0])
                for idx, line in enumerate(open(dict_path, "r").readlines()))


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def infer(args):
    word = fluid.layers.data(name='word', shape=[1], dtype='int64', lod_level=1)
    mention = fluid.layers.data(
        name='mention', shape=[1], dtype='int64', lod_level=1)
    target = fluid.layers.data(
        name='target', shape=[1], dtype='int64', lod_level=1)

    label_reverse_dict = load_reverse_dict(args.test_label_file)

    test_data = paddle.batch(
        reader.file_reader(args.test_data_dir), batch_size=args.batch_size)
    place = fluid.CUDAPlace(0) if args.device == 'GPU' else fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[word, mention, target], place=place)
    exe = fluid.Executor(place)

    inference_scope = fluid.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(args.model_path, exe)
        total_passes = args.num_passes + args.skip_pass_num
        batch_times = [0] * total_passes
        word_counts = [0] * total_passes
        wpses = [0] * total_passes
        all_iters = 0
        for pass_id in range(total_passes):
            if pass_id < args.skip_pass_num:
                print("Warm-up pass")
            if pass_id == args.skip_pass_num:
                profiler.reset_profiler()
            iters = 0
            for data in test_data():
                word = to_lodtensor(list(map(lambda x: x[0], data)), place)
                mention = to_lodtensor(list(map(lambda x: x[1], data)), place)

                start = time.time()
                crf_decode = exe.run(inference_program,
                                     feed={"word": word,
                                           "mention": mention},
                                     fetch_list=fetch_targets,
                                     return_numpy=False)
                batch_time = time.time() - start
                lod_info = (crf_decode[0].lod())[0]
                np_data = np.array(crf_decode[0])
                word_count = 0
                assert len(data) == len(lod_info) - 1
                for sen_index in range(len(data)):
                    assert len(data[sen_index][0]) == lod_info[
                        sen_index + 1] - lod_info[sen_index]
                    word_index = 0
                    for tag_index in range(lod_info[sen_index],
                                           lod_info[sen_index + 1]):
                        word = str(data[sen_index][0][word_index])
                        gold_tag = label_reverse_dict[data[sen_index][2][
                            word_index]]
                        tag = label_reverse_dict[np_data[tag_index][0]]
                        word_index += 1
                    word_count += word_index
                batch_times[pass_id] += batch_time
                word_counts[pass_id] += word_count
                iters += 1
                all_iters += 1
            batch_times[pass_id] /= iters
            word_counts[pass_id] /= iters
            wps = word_counts[pass_id] / batch_times[pass_id]
            wpses[pass_id] = wps

            print(
                "Pass: %d, iterations (total): %d (%d), latency: %.5f s, words: %d, wps: %f"
                % (pass_id, iters, all_iters, batch_times[pass_id],
                   word_counts[pass_id], wps))

    # Postprocess benchmark data
    latencies = batch_times[args.skip_pass_num:]
    latency_avg = np.average(latencies)
    latency_std = np.std(latencies)
    latency_pc99 = np.percentile(latencies, 99)
    wps_avg = np.average(wpses)
    wps_std = np.std(wpses)
    wps_pc01 = np.percentile(wpses, 1)

    # Benchmark output
    print('\nTotal passes (incl. warm-up): %d' % (total_passes))
    print('Total iterations (incl. warm-up): %d' % (all_iters))
    print('Total examples (incl. warm-up): %d' % (all_iters * args.batch_size))
    print('avg latency: %.5f, std latency: %.5f, 99pc latency: %.5f' %
          (latency_avg, latency_std, latency_pc99))
    print('avg wps: %.5f, std wps: %.5f, wps for 99pc latency: %.5f' %
          (wps_avg, wps_std, wps_pc01))


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    if args.profile:
        if args.device == 'GPU':
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                infer(args)
        else:
            with profiler.profiler('CPU', sorted_key='total') as cpuprof:
                infer(args)
    else:
        infer(args)
