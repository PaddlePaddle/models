import os
import shutil
import sys

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
TOOLS_PATH = os.path.join(LOCAL_PATH, "..", "..", "tools")
sys.path.append(TOOLS_PATH)

from tools import download_file_and_uncompress

if __name__ == '__main__':
    url = "https://paddlerec.bj.bcebos.com/deepfm%2Fdist_data_demo.tar.gz"

    print("download and extract starting...")
    download_file_and_uncompress(url, savename="dist_data_demo.tar.gz")
    print("download and extract finished")

    print("preprocessing...")
    os.system("python preprocess_dist.py")
    print("preprocess done")

    print("done")