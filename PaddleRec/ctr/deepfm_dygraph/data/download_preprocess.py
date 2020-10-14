import os
import shutil
import sys

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
TOOLS_PATH = os.path.join(LOCAL_PATH, "..", "..", "tools")
sys.path.append(TOOLS_PATH)

from tools import download_file_and_uncompress, download_file

if __name__ == '__main__':
    url = "https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz"
    url2 = "https://paddlerec.bj.bcebos.com/deepfm%2Ffeat_dict_10.pkl2"

    print("download and extract starting...")
    download_file_and_uncompress(url)
    if not os.path.exists("aid_data"):
        os.makedirs("aid_data")
    download_file(url2, "./aid_data/feat_dict_10.pkl2", True)
    print("download and extract finished")

    print("preprocessing...")
    os.system("python preprocess.py")
    print("preprocess done")

    shutil.rmtree("raw_data")
    print("done")
