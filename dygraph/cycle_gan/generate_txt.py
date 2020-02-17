import sys
import os
import argparse

def gen_txt(dir_path):
    dataname = "cityscapes"
    ### generator .txt file according to dirs
    dirs = os.listdir(os.path.join(dir_path, '{}'.format(dataname)))
    for d in dirs:
        txt_file = d + '.txt'
        txt_dir = os.path.join(dir_path, dataname)
        f = open(os.path.join(txt_dir, txt_file), 'w')
        for fil in os.listdir(os.path.join(txt_dir, d)):
            wl = d + '/' + fil + '\n'
            f.write(wl)
        f.close()
    sys.stderr.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # yapf: disable
    parser.add_argument('--output_dir',       type=str,     default="datasets",       help='Path to output Cityscapes directory.')
    # yapf: enable
    args = parser.parse_args()
    gen_txt(args.output_dir)