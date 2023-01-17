import os
import sys
import time

import requests
import zipfile

FLUSH_INTERVAL = 0.1
lasttime = time.time()


def progress(str, end=False):
    global lasttime
    if end:
        str += "\n"
        lasttime = 0
    if time.time() - lasttime >= FLUSH_INTERVAL:
        sys.stdout.write("\r%s" % str)
        lasttime = time.time()
        sys.stdout.flush()


def download_file(url, savepath, print_progress=True):
    if print_progress:
        print("Connecting to {}".format(url))
    r = requests.get(url, stream=True, timeout=15)
    total_length = r.headers.get('content-length')

    if total_length is None:
        with open(savepath, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    else:
        with open(savepath, 'wb') as f:
            dl = 0
            total_length = int(total_length)
            starttime = time.time()
            if print_progress:
                print("Downloading %s" % os.path.basename(savepath))
            for data in r.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                if print_progress:
                    done = int(50 * dl / total_length)
                    progress("[%-50s] %.2f%%" %
                             ('=' * done, float(100 * dl) / total_length))
        if print_progress:
            progress("[%-50s] %.2f%%" % ('=' * 50, 100), end=True)


def uncompress(path):
    files = zipfile.ZipFile(path, 'r')
    filelist = files.namelist()
    rootpath = filelist[0]
    for file in filelist:
        files.extract(file, './')
