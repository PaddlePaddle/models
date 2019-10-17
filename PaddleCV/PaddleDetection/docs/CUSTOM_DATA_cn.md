# 如何添加自定义数据集与预处理


## 添加数据集

PaddleDetection支持两种数据集， `Indexable` （适合本地文件系统）和 `Iterable` (流数据集，适合网络文件系统）。

Indexable数据集可以类比Python的 `Sequence` ，需要实现 `__len__` 和 `__getitem__` 方法。

```python
from ppdet.data import DataSet
from glob import glob
import os


class MyFolderDataset(DataSet):
    def __init__(self, root_dir):
        super(MyFolderDataset, self).__init__()
        self.image_files = glob(os.path.join(root_dir, '*.jpg'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_file[idx]
        # `_read_image` is inherited from DataSet
        img = self._read_image(sample['file'])
        return {'image': img}
```

也可以使用 `ppdet.data.ListDataSet` 并指定 `list_fn` 方法，请参看 `ImageFolder` 数据集实现

```python
class ImageFolder(ListDataSet):
    def __init__(self, root_dir=None):
        def find_image_files(dir):
            image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.ppm', '.pgm']

            samples = []
            for target in sorted(os.listdir(dir)):
                path = os.path.join(dir, target)
                if not os.path.isfile(path):
                    continue
                if os.path.splitext(target)[1].lower() in image_exts:
                    samples.append({'file': path})

            return samples
        super(ImageFolder, self).__init__(find_image_files, root_dir)
```

`Iterable` 数据集可以类比Python的 `Generator` ，需要实现 `__next__` 方法

```python
from ppdet.data import DataSet
from glob import glob
import os


class MyFolderDataset(DataSet):
    def __init__(self, root_dir):
        super(MyFolderDataset, self).__init__()
        image_files = glob(os.path.join(root_dir, '*.jpg'))
        self._iter = iter(image_files)

    def __next__(self):
        image_file = next(self._iter)
        # `_read_image` is inherited from DataSet
        img = self._read_image(sample['file'])
        return {'image': img}

    # python2 compatibility
    next = __next__
```

也可以使用 `ppdet.data.StreamDataSet` 并提供 `generator` 实例。比如：

```python
from ppdet.data import StreamDataSet


class ImageFolder(StreamDataSet):
    def __init__(self, root_dir=None):
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.ppm', '.pgm']

        samples = []
        for target in sorted(os.listdir(dir)):
            path = os.path.join(dir, target)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(target)[1].lower() in image_exts:
                samples.append({'file': path})

        super(ImageFolder, self).__init__(iter(samples))
```

上述两种数据集的 `__getitem__` 和 `__next__` 需返回 Python dict作为一个sample。每个sample包含以下字段：

-   `image`: 所有模型均需要，输入的图像数据。
-   `gt_box`, `gt_label`: 所有模型训练均需要。bbox的ground truth。
-   `gt_poly`: Mask-RCNN系模型需要，mask的ground truth。
-   `width`, `height`: 大部分模型需要用到，图像的维度，预处理步骤可能会更改该字段的值。
-   `orig_width`, `orig_height`: 所有模型均需要用到，图像维度的原始值，数据集返回值和 width, height 相同即可。
-   `scale`: 适配RCNN等模型，图像的缩放比例，预处理步骤可能会更改该字段的值，数据集返回1即可。
-   `id`: 类似mscoco数据集评估时需要用到。
-   `is_crowd`: 类似mscoco数据集评估需要用到。
-   `difficult`: 类似voc的数据集评估需要用到。

用户也可以根据自己需要添加其他字段。


## 添加预处理

PaddleDetection的预处理分为两个步骤，sample transform会在sample级别进行处理，batch transform会处理已经组好batch的数据，如将整个batch的图像pad到相同大小。

预处理一般使用Python callable class， 需实现 `__call__(self, input)` 方法 （ `input` 可以是一个sample也可以是一个batch，均为python dict）。在该方法中，可以对输入的dict各字段进行操作，也可以根据需要添加新字段。

```python
import cv2
from ppdet.core.workspace import register, serializable


@register
@serializable
class MyResizeImage(object):
    def __init__(self, size):
        super(MyResizeImage, self).__init__()
        self.size = size

    def __call__(self, sample):
        sample['image'] = cv2.resize(sample['image'], (self.size, self.size))
        # custom data field
        sample['my_size_h'] = self.size
        sample['my_size_w'] = self.size
```


## 自定义数据集/预处理的相应配置

自定义数据集在使用前需要先调用 `ppdet.data.register_dataset` 方法进行注册，然后即可根据注册的名字配置。

```python
from ppdet.data import register_dataset

register_dataset('myfolder', MyFolderDataset)
```

自定义预处理步骤需要添加到相应 `sample_transforms` 或 `batch_transforms` 配置项中。

自定义字段可以作为 `feed_var` 实际传入网络，如果是不要传入网络但会在eval或test等中用到的其他字段也可以作为 `extra_var` 传入。 `feed_var` 和 `extra_var` 支持一定程度的组合搭配。

```yaml
TrainDataLoader:
  batch_size: 1
  dataset:
    type: myfolder
    root_dir: /data
  sample_transforms:
  - !MyResizeImage
    size: 300
  feed_vars:
  - image
  # combined 2 fields into a single var
  - name: im_size
    fields:
    - my_size_h
    - my_size_w
  extra_vars:
  - my_size_w
```
