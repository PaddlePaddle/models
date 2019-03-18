from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing
cpus = multiprocessing.cpu_count()
cpus = min(36,cpus)


PATH = Path('/data/imagenet2')
DEST = Path('/data/imagenet2/sz')
def mkdir(path):
    if not path.exists():
        path.mkdir()

mkdir(DEST)
szs = (160, 352)

def resize_img(p, im, fn, sz):
    w,h = im.size
    ratio = min(h/sz,w/sz)
    im = im.resize((int(w/ratio), int(h/ratio)), resample=Image.BICUBIC)
    new_fn = DEST/str(sz)/fn.relative_to(PATH)
    mkdir(new_fn.parent())
    im.save(new_fn)

def resizes(p, fn):
    im = Image.open(fn)
    for sz in szs: resize_img(p, im, fn, sz)

def resize_imgs(p):
    files = p.glob('*/*.jpeg')
    with ProcessPoolExecutor(cpus) as e: e.map(partial(resizes, p), files)


for sz in szs:
    ssz=str(sz)
    mkdir((DEST/ssz))
    for ds in ('validation','train'): mkdir((DEST/ssz/ds))
    for ds in ('train',): mkdir((DEST/ssz/ds))

for ds in ("validation", "train"):
    print(PATH/ds)
    resize_imgs(PATH/ds)