import argparse
import sys
import os
import os.path as osp
import glob
from pipes import quote
from multiprocessing import Pool, current_process
import cv2


def dump_frames(vid_item):
    full_path, vid_path, vid_id = vid_item
    vid_name = vid_path.split('.')[0]
    out_full_path = osp.join(args.out_dir, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    vr = cv2.VideoCapture(full_path)
    videolen = int(vr.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(videolen):
        ret, frame = vr.read()
        if ret == False:
            continue
        img = frame[:, :, ::-1]
        # covert the BGR img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img is not None:
            # cv2.imwrite will write BGR into RGB images
            cv2.imwrite('{}/img_{:05d}.jpg'.format(out_full_path, i + 1), img)
        else:
            print('[Warning] length inconsistent!'
                  'Early stop with {} out of {} frames'.format(i + 1, videolen))
            break
    print('{} done with {} frames'.format(vid_name, videolen))
    sys.stdout.flush()
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='extract frames')
    parser.add_argument('src_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--level', type=int, choices=[1, 2], default=2)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument(
        "--ext",
        type=str,
        default='avi',
        choices=['avi', 'mp4'],
        help='video file extensions')

    parser.add_argument(
        "--resume",
        action='store_true',
        default=False,
        help='resume optical flow extraction '
        'instead of overwriting')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    if not osp.isdir(args.out_dir):
        print('Creating folder: {}'.format(args.out_dir))
        os.makedirs(args.out_dir)
    if args.level == 2:
        classes = os.listdir(args.src_dir)
        for classname in classes:
            new_dir = osp.join(args.out_dir, classname)
            if not osp.isdir(new_dir):
                print('Creating folder: {}'.format(new_dir))
                os.makedirs(new_dir)

    print('Reading videos from folder: ', args.src_dir)
    print('Extension of videos: ', args.ext)
    if args.level == 2:
        fullpath_list = glob.glob(args.src_dir + '/*/*.' + args.ext)
        done_fullpath_list = glob.glob(args.out_dir + '/*/*')
    elif args.level == 1:
        fullpath_list = glob.glob(args.src_dir + '/*.' + args.ext)
        done_fullpath_list = glob.glob(args.out_dir + '/*')
    print('Total number of videos found: ', len(fullpath_list))
    if args.resume:
        fullpath_list = set(fullpath_list).difference(set(done_fullpath_list))
        fullpath_list = list(fullpath_list)
        print('Resuming. number of videos to be done: ', len(fullpath_list))

    if args.level == 2:
        vid_list = list(
            map(lambda p: osp.join('/'.join(p.split('/')[-2:])), fullpath_list))
    elif args.level == 1:
        vid_list = list(map(lambda p: p.split('/')[-1], fullpath_list))

    pool = Pool(args.num_worker)
    pool.map(dump_frames, zip(fullpath_list, vid_list, range(len(vid_list))))
