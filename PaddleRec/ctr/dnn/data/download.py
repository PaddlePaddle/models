import os
import shutil
import sys
import glob

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
TOOLS_PATH = os.path.join(LOCAL_PATH, "..", "..", "tools")
sys.path.append(TOOLS_PATH)

from tools import download_file_and_uncompress

if __name__ == '__main__':
    url = "https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz"

    print("download and extract starting...")
    download_file_and_uncompress(url)
    print("download and extract finished")

    if os.path.exists("raw"):
        shutil.rmtree("raw")
    os.mkdir("raw")

    # mv ./*.txt raw/
    files = glob.glob("*.txt")
    for f in files:
        shutil.move(f, "raw")

    print("done")
