import os
import time
import shutil
import requests
import sys
import tarfile
import zipfile
import platform
import functools

lasttime = time.time()
FLUSH_INTERVAL = 0.1

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))


def get_platform():
    return platform.platform()


def is_windows():
    return get_platform().lower().startswith("windows")


def progress(str, end=False):
    global lasttime
    if end:
        str += "\n"
        lasttime = 0
    if time.time() - lasttime >= FLUSH_INTERVAL:
        sys.stdout.write("\r%s" % str)
        lasttime = time.time()
        sys.stdout.flush()


def download_file(url, savepath, print_progress):
    r = requests.get(url, stream=True)
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


def _uncompress_file(filepath, extrapath, delete_file, print_progress):
    if print_progress:
        print("Uncompress %s" % os.path.basename(filepath))

    if filepath.endswith("zip"):
        handler = _uncompress_file_zip
    elif filepath.endswith("tgz"):
        handler = _uncompress_file_tar
    else:
        handler = functools.partial(_uncompress_file_tar, mode="r")

    for total_num, index, rootpath in handler(filepath, extrapath):
        if print_progress:
            done = int(50 * float(index) / total_num)
            progress("[%-50s] %.2f%%" %
                     ('=' * done, float(100 * index) / total_num))
    if print_progress:
        progress("[%-50s] %.2f%%" % ('=' * 50, 100), end=True)

    if delete_file:
        os.remove(filepath)

    return rootpath


def _uncompress_file_zip(filepath, extrapath):
    files = zipfile.ZipFile(filepath, 'r')
    filelist = files.namelist()
    rootpath = filelist[0]
    total_num = len(filelist)
    for index, file in enumerate(filelist):
        files.extract(file, extrapath)
        yield total_num, index, rootpath
    files.close()
    yield total_num, index, rootpath


def _uncompress_file_tar(filepath, extrapath, mode="r:gz"):
    files = tarfile.open(filepath, mode)
    filelist = files.getnames()
    total_num = len(filelist)
    rootpath = filelist[0]
    for index, file in enumerate(filelist):
        files.extract(file, extrapath)
        yield total_num, index, rootpath
    files.close()
    yield total_num, index, rootpath


def download_file_and_uncompress(url,
                                 savepath=None,
                                 savename=None,
                                 extrapath=None,
                                 print_progress=True,
                                 cover=False,
                                 delete_file=False):
    if savepath is None:
        savepath = "."

    if extrapath is None:
        extrapath = "."

    if savename is None:
        savename = url.split("/")[-1]
    savepath = os.path.join(savepath, savename)

    if cover:
        if os.path.exists(savepath):
            shutil.rmtree(savepath)

    if not os.path.exists(savepath):
        download_file(url, savepath, print_progress)
    _ = _uncompress_file(savepath, extrapath, delete_file, print_progress)
