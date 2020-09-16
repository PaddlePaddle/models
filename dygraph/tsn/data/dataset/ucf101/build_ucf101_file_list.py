import argparse
import os
import glob
import fnmatch
import random


def parse_directory(path,
                    key_func=lambda x: x[-11:],
                    rgb_prefix='img_',
                    level=1):
    """
    Parse directories holding extracted frames from standard benchmarks
    """
    print('parse frames under folder {}'.format(path))
    if level == 1:
        frame_folders = glob.glob(os.path.join(path, '*'))
    elif level == 2:
        frame_folders = glob.glob(os.path.join(path, '*', '*'))
    else:
        raise ValueError('level can be only 1 or 2')

    def count_files(directory, prefix_list):
        lst = os.listdir(directory)
        cnt_list = [len(fnmatch.filter(lst, x + '*')) for x in prefix_list]
        return cnt_list

    # check RGB
    frame_dict = {}
    for i, f in enumerate(frame_folders):
        all_cnt = count_files(f, (rgb_prefix))
        k = key_func(f)

        x_cnt = all_cnt[1]
        y_cnt = all_cnt[2]
        if x_cnt != y_cnt:
            raise ValueError('x and y direction have different number '
                             'of flow images. video: ' + f)
        if i % 200 == 0:
            print('{} videos parsed'.format(i))

        frame_dict[k] = (f, all_cnt[0], x_cnt)

    print('frame folder analysis done')
    return frame_dict


def build_split_list(split, frame_info, shuffle=False):
    def build_set_list(set_list):
        rgb_list = list()
        for item in set_list:
            if item[0] not in frame_info:
                continue
            elif frame_info[item[0]][1] > 0:
                rgb_cnt = frame_info[item[0]][1]
                rgb_list.append('{} {} {}\n'.format(item[0], rgb_cnt, item[1]))
            else:
                rgb_list.append('{} {}\n'.format(item[0], item[1]))
        if shuffle:
            random.shuffle(rgb_list)
        return rgb_list

    train_rgb_list = build_set_list(split[0])
    test_rgb_list = build_set_list(split[1])
    return (train_rgb_list, test_rgb_list)


def parse_ucf101_splits(level):
    class_ind = [x.strip().split() for x in open('./annotations/classInd.txt')]
    class_mapping = {x[1]: int(x[0]) - 1 for x in class_ind}

    def line2rec(line):
        items = line.strip().split(' ')
        vid = items[0].split('.')[0]
        vid = '/'.join(vid.split('/')[-level:])
        label = class_mapping[items[0].split('/')[0]]
        return vid, label

    splits = []
    for i in range(1, 4):
        train_list = [
            line2rec(x)
            for x in open('./annotations/trainlist{:02d}.txt'.format(i))
        ]
        test_list = [
            line2rec(x)
            for x in open('./annotations/testlist{:02d}.txt'.format(i))
        ]
        splits.append((train_list, test_list))
    return splits


def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument(
        'frame_path', type=str, help='root directory for the frames')
    parser.add_argument('--rgb_prefix', type=str, default='img_')
    parser.add_argument('--num_split', type=int, default=3)
    parser.add_argument('--level', type=int, default=2, choices=[1, 2])
    parser.add_argument(
        '--format',
        type=str,
        default='rawframes',
        choices=['rawframes', 'videos'])
    parser.add_argument('--out_list_path', type=str, default='./')
    parser.add_argument('--shuffle', action='store_true', default=False)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.level == 2:

        def key_func(x):
            return '/'.join(x.split('/')[-2:])
    else:

        def key_func(x):
            return x.split('/')[-1]

    if args.format == 'rawframes':
        frame_info = parse_directory(
            args.frame_path,
            key_func=key_func,
            rgb_prefix=args.rgb_prefix,
            level=args.level)
    elif args.format == 'videos':
        if args.level == 1:
            video_list = glob.glob(os.path.join(args.frame_path, '*'))
        elif args.level == 2:
            video_list = glob.glob(os.path.join(args.frame_path, '*', '*'))
        frame_info = {
            os.path.relpath(x.split('.')[0], args.frame_path): (x, -1, -1)
            for x in video_list
        }

    split_tp = parse_ucf101_splits(args.level)
    assert len(split_tp) == args.num_split

    out_path = args.out_list_path

    for i, split in enumerate(split_tp):
        lists = build_split_list(split_tp[i], frame_info, shuffle=args.shuffle)
        filename = 'ucf101_train_split_{}_{}.txt'.format(i + 1, args.format)

        PATH = os.path.abspath(args.frame_path)
        with open(os.path.join(out_path, filename), 'w') as f:
            f.writelines([os.path.join(PATH, item) for item in lists[0]])
        filename = 'ucf101_val_split_{}_{}.txt'.format(i + 1, args.format)
        with open(os.path.join(out_path, filename), 'w') as f:
            f.writelines([os.path.join(PATH, item) for item in lists[1]])


if __name__ == "__main__":
    main()
