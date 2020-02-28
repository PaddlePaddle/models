import os
import argparse
import functools
import glob
from PIL import Image
''' Based on https://github.com/junyanz/CycleGAN'''


def load_image(path):
    return Image.open(path).convert('RGB').resize((256, 256))


def propress_cityscapes(gtFine_dir, leftImg8bit_dir, output_dir, phase):
    save_dir = os.path.join(output_dir, phase)
    try:
        os.makedirs(save_dir)
    except Exception as e:
        print("{} makedirs".format(e))
        pass
    try:
        os.makedirs(os.path.join(save_dir, 'A'))
    except Exception as e:
        print("{} makedirs".format(e))
    try:
        os.makedirs(os.path.join(save_dir, 'B'))
    except Exception as e:
        print("{} makedirs".format(e))

    seg_expr = os.path.join(gtFine_dir, phase, "*", "*_color.png")
    seg_paths = glob.glob(seg_expr)
    seg_paths = sorted(seg_paths)

    photo_expr = os.path.join(leftImg8bit_dir, phase, "*", '*_leftImg8bit.png')
    photo_paths = glob.glob(photo_expr)
    photo_paths = sorted(photo_paths)

    assert len(seg_paths) == len(photo_paths), \
          "[%d] gtFine images NOT match [%d] leftImg8bit images. Aborting." % (len(segmap_paths), len(photo_paths))

    for i, (seg_path, photo_path) in enumerate(zip(seg_paths, photo_paths)):
        seg_image = load_image(seg_path)
        photo_image = load_image(photo_path)
        # save image
        save_path = os.path.join(save_dir, 'A', "%d_A.jpg" % i)
        photo_image.save(save_path, format='JPEG', subsampling=0, quality=100)
        save_path = os.path.join(save_dir, 'B', "%d_B.jpg" % i)
        seg_image.save(save_path, format='JPEG', subsampling=0, quality=100)

        if i % 10 == 0:
            print("proprecess %d ~ %d images." % (i, i + 10))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    # yapf: disable
    parser.add_argument('--gtFine_dir',       type=str,     default=None,       help='Path to Cityscapes gtFine directory.')
    parser.add_argument('--leftImg8bit_dir',  type=str,     default=None,       help='Path to Cityscapes leftImg8bit_trainvaltest directory.')
    parser.add_argument('--output_dir',       type=str,     default=None,       help='Path to output Cityscapes directory.')
    # yapf: enable
    args = parser.parse_args()

    print('Preparing Cityscapes Dataset for val phase')
    propress_cityscapes(args.gtFine_dir, args.leftImg8bit_dir, args.output_dir,
                        'val')

    print('Preparing Cityscapes Dataset for train phase')
    propress_cityscapes(args.gtFine_dir, args.leftImg8bit_dir, args.output_dir,
                        'train')

    print("DONE!!!")
