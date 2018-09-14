import cv2
import numpy as np

default_config = {
    "shuffle": True,
    "min_resize": 0.5,
    "max_resize": 2,
    "crop_size": 769,
}

def slice_with_pad(a, s, value=0):
    pads = []
    slices = []
    for i in range(len(a.shape)):
        if i >= len(s):
            pads.append([0,0])
            slices.append([0, a.shape[i]])
        else:
            l,r = s[i]
            if l<0:
                pl = -l
                l = 0
            else:
                pl = 0
            if r>a.shape[i]:
                pr = r - a.shape[i]
                r = a.shape[i]
            else:
                pr = 0
            pads.append([pl,pr])
            slices.append([l,r])
    slices = map(lambda x: slice(x[0],x[1],1), slices)
    a = a[slices]
    a = np.pad(a, pad_width=pads, mode='constant', constant_values=value)
    return a

class Cityscape_dataset:
    def __init__(self, subset='train', dataset_dir='/home/cjld/nfs/liangdun/cityscape-data/', config=default_config):
        import commands
        label_dirname = dataset_dir + 'gtFine/' + subset
        label_files = commands.getoutput("find %s -type f | grep labelTrainIds | sort" % label_dirname).splitlines()
        # label_files = !find $label_dirname -type f | grep labelTrainIds
        self.label_files = label_files
        self.label_dirname = label_dirname
        self.index = 0
        self.subset = subset
        self.dataset_dir = dataset_dir
        self.config = config
        self.reset()
        print "total number", len(label_files)

    def reset(self, shuffle=False):
        self.index = 0
        if self.config["shuffle"]:
            np.random.shuffle(self.label_files)

    def next_img(self):
        self.index += 1
        if self.index >= len(self.label_files):
            self.reset()

    def get_img(self):
        shape = self.config["crop_size"]
        while True:
            ln = self.label_files[self.index]
            img_name = self.dataset_dir + 'leftImg8bit/' + self.subset + ln[len(self.label_dirname):]
            img_name = img_name.replace('gtFine_labelTrainIds', 'leftImg8bit')
            label = cv2.imread(ln)
            img = cv2.imread(img_name)
            if img is None:
                print "load img failed:", img_name
                self.next_img()
            else:
                break
        if shape == -1:
            return img, label, ln
        random_scale = np.random.rand(1)*(self.config['max_resize'] - self.config['min_resize']) + self.config['min_resize']
        crop_size = int(shape/random_scale)
        bb = crop_size//2
        def my_randint(low, high):
            return int(np.random.rand(1) * (high-low)+low)
        offset_x = np.random.randint(bb, max(bb+1,img.shape[0]-bb)) - crop_size//2
        offset_y = np.random.randint(bb, max(bb+1,img.shape[1]-bb)) - crop_size//2
        img_crop = slice_with_pad(img, [[offset_x,offset_x+crop_size],[offset_y,offset_y+crop_size]], 128)
        img = cv2.resize(img_crop, (shape,shape))
        label_crop = slice_with_pad(label, [[offset_x,offset_x+crop_size],[offset_y,offset_y+crop_size]], 255)
        label = cv2.resize(label_crop, (shape,shape), interpolation=cv2.INTER_NEAREST)
        return img, label, ln+str((offset_x, offset_y, crop_size, random_scale))

    def get_batch(self, batch_size=1):
        imgs = []
        labels = []
        names = []
        while len(imgs) < batch_size:
            img, label, ln = self.get_img()
            imgs.append(img)
            labels.append(label)
            names.append(ln)
            self.next_img()
        return np.array(imgs), np.array(labels), names

#dataset = Cityscape_dataset()

import contextlib

def get_handle():
    import IPython
    handle = IPython.display.DisplayHandle()
    handle.display('wait')
    return handle

@contextlib.contextmanager
def myfig_display(figsize=(8,8), h=None):
    import pylab as pl
    if h == None: h = handle
    fig = pl.figure(figsize=figsize)
    try:
        yield fig
    finally:
        h.update(fig)
        pl.close(fig)
