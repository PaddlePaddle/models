# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division

import os

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.paddle import DALIGenericIterator

import paddle
from paddle import fluid


class HybridTrainPipe(Pipeline):
    def __init__(self,
                 file_root,
                 file_list,
                 batch_size,
                 resize_shorter,
                 crop,
                 min_area,
                 lower,
                 upper,
                 interp,
                 mean,
                 std,
                 device_id,
                 shard_id=0,
                 num_shards=1,
                 random_shuffle=True,
                 num_threads=4,
                 seed=42):
        super(HybridTrainPipe, self).__init__(
            batch_size, num_threads, device_id, seed=seed)
        self.input = ops.FileReader(
            file_root=file_root,
            file_list=file_list,
            shard_id=shard_id,
            num_shards=num_shards,
            random_shuffle=random_shuffle)
        # set internal nvJPEG buffers size to handle full-sized ImageNet images
        # without additional reallocations
        device_memory_padding = 211025920
        host_memory_padding = 140544512
        self.decode = ops.ImageDecoderRandomCrop(
            device='mixed',
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
            random_aspect_ratio=[lower, upper],
            random_area=[min_area, 1.0],
            num_attempts=100)
        self.res = ops.Resize(
            device='gpu', resize_x=crop, resize_y=crop, interp_type=interp)
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop=(crop, crop),
            image_type=types.RGB,
            mean=mean,
            std=std)
        self.coin = ops.CoinFlip(probability=0.5)
        self.to_int64 = ops.Cast(dtype=types.INT64, device="gpu")

    def define_graph(self):
        rng = self.coin()
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.to_int64(labels.gpu())]

    def __len__(self):
        return self.epoch_size("Reader")


class HybridValPipe(Pipeline):
    def __init__(self,
                 file_root,
                 file_list,
                 batch_size,
                 resize_shorter,
                 crop,
                 interp,
                 mean,
                 std,
                 device_id,
                 shard_id=0,
                 num_shards=1,
                 random_shuffle=False,
                 num_threads=4,
                 seed=42):
        super(HybridValPipe, self).__init__(
            batch_size, num_threads, device_id, seed=seed)
        self.input = ops.FileReader(
            file_root=file_root,
            file_list=file_list,
            shard_id=shard_id,
            num_shards=num_shards,
            random_shuffle=random_shuffle)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(
            device="gpu", resize_shorter=resize_shorter, interp_type=interp)
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop=(crop, crop),
            image_type=types.RGB,
            mean=mean,
            std=std)
        self.to_int64 = ops.Cast(dtype=types.INT64, device="gpu")

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.to_int64(labels.gpu())]

    def __len__(self):
        return self.epoch_size("Reader")


def build(settings, mode='train'):
    env = os.environ
    assert settings.use_gpu, "gpu training is required for DALI"
    assert not settings.use_mixup, "mixup is not supported by DALI reader"
    assert not settings.use_aa, "auto augment is not supported by DALI reader"
    assert float(env.get('FLAGS_fraction_of_gpu_memory_to_use', 0.92)) < 0.9, \
        "Please leave enough GPU memory for DALI workspace, e.g., by setting" \
        " `export FLAGS_fraction_of_gpu_memory_to_use=0.8`"

    file_root = settings.data_dir
    bs = settings.batch_size
    assert bs % paddle.fluid.core.get_cuda_device_count() == 0, \
        "batch size must be multiple of number of devices"
    batch_size = bs // paddle.fluid.core.get_cuda_device_count()

    mean = [v * 255 for v in settings.image_mean]
    std = [v * 255 for v in settings.image_std]
    crop = settings.image_shape[1]
    resize_shorter = settings.resize_short_size
    min_area = settings.lower_scale
    lower = settings.lower_ratio
    upper = settings.upper_ratio

    interp = settings.interpolation or 1  # default to linear
    interp_map = {
        0: types.INTERP_NN,  # cv2.INTER_NEAREST
        1: types.INTERP_LINEAR,  # cv2.INTER_LINEAR
        2: types.INTERP_CUBIC,  # cv2.INTER_CUBIC
        4: types.INTERP_LANCZOS3,  # XXX use LANCZOS3 for cv2.INTER_LANCZOS4
    }
    assert interp in interp_map, "interpolation method not supported by DALI"
    interp = interp_map[interp]

    if mode != 'train':
        p = fluid.framework.cuda_places()[0]
        place = fluid.core.Place()
        place.set_place(p)
        device_id = place.gpu_device_id()
        file_list = os.path.join(file_root, 'val_list.txt')
        if not os.path.exists(file_list):
            file_list = None
            file_root = os.path.join(file_root, 'val')
        pipe = HybridValPipe(
            file_root,
            file_list,
            batch_size,
            resize_shorter,
            crop,
            interp,
            mean,
            std,
            device_id=device_id)
        pipe.build()
        return DALIGenericIterator(
            pipe, ['feed_image', 'feed_label'],
            size=len(pipe),
            dynamic_shape=True,
            fill_last_batch=True,
            last_batch_padded=True)

    file_list = os.path.join(file_root, 'train_list.txt')
    if not os.path.exists(file_list):
        file_list = None
        file_root = os.path.join(file_root, 'train')

    if 'PADDLE_TRAINER_ID' in env and 'PADDLE_TRAINERS_NUM' in env:
        shard_id = int(env['PADDLE_TRAINER_ID'])
        num_shards = int(env['PADDLE_TRAINERS_NUM'])
        device_id = int(env['FLAGS_selected_gpus'])
        pipe = HybridTrainPipe(
            file_root,
            file_list,
            batch_size,
            resize_shorter,
            crop,
            min_area,
            lower,
            upper,
            interp,
            mean,
            std,
            device_id,
            shard_id,
            num_shards,
            seed=42 + shard_id)
        pipe.build()
        pipelines = [pipe]
        sample_per_shard = len(pipe) // num_shards
    else:
        pipelines = []
        places = fluid.framework.cuda_places()
        num_shards = len(places)
        for idx, p in enumerate(places):
            place = fluid.core.Place()
            place.set_place(p)
            device_id = place.gpu_device_id()
            pipe = HybridTrainPipe(
                file_root,
                file_list,
                batch_size,
                resize_shorter,
                crop,
                min_area,
                lower,
                upper,
                interp,
                mean,
                std,
                device_id,
                idx,
                num_shards,
                seed=42 + idx)
            pipe.build()
            pipelines.append(pipe)
        sample_per_shard = len(pipelines[0])

    return DALIGenericIterator(
        pipelines, ['feed_image', 'feed_label'], size=sample_per_shard)


def train(settings):
    return build(settings, 'train')


def val(settings):
    return build(settings, 'val')
