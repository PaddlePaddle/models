import os
import sys
import io

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
TOOLS_PATH = os.path.join(LOCAL_PATH, "..", "..", "tools")
sys.path.append(TOOLS_PATH)

from tools import download_file_and_uncompress

if __name__ == '__main__':
    trainfile = 'train.txt'
    url = "https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz"

    print("download and extract starting...")
    download_file_and_uncompress(url)
    print("download and extract finished")

    count = 0
    for _ in io.open(trainfile, 'r', encoding='utf-8'):
        count += 1

    print("total records: %d" % count)
    print("done")
