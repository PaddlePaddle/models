import os
import shutil
import sys

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
TOOLS_PATH = os.path.join(LOCAL_PATH, "..", "..", "tools")
sys.path.append(TOOLS_PATH)

from tools import download_file_and_uncompress, download_file

if __name__ == '__main__':
    url_train = "https://paddlerec.bj.bcebos.com/xdeepfm%2Ftr"
    url_test = "https://paddlerec.bj.bcebos.com/xdeepfm%2Fev"

    train_dir = "train_data"
    test_dir = "test_data"

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    print("download and extract starting...")
    download_file(url_train, "./train_data/tr", True)
    download_file(url_test, "./test_data/ev", True)
    print("download and extract finished")

    print("done")
