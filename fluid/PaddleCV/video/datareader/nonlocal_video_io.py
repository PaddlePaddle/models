#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import cv2
import numpy as np
import random


def video_fast_get_frame(video_path,
                         sampling_rate=1,
                         length=64,
                         start_frm=-1,
                         sample_times=1):
    cap = cv2.VideoCapture(video_path)
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sampledFrames = []

    # n_frame <  sample area
    video_output = np.ndarray(shape=[length, height, width, 3], dtype=np.uint8)

    use_start_frm = start_frm
    if start_frm < 0:
        if (frame_cnt - length * sampling_rate > 0):
            use_start_frm = random.randint(0,
                                           frame_cnt - length * sampling_rate)
        else:
            use_start_frm = 0
    else:
        frame_gaps = float(frame_cnt) / float(sample_times)
        use_start_frm = int(frame_gaps * start_frm) % frame_cnt

    for i in range(frame_cnt):
        ret, frame = cap.read()
        # maybe first frame is empty
        if ret == False:
            continue
        img = frame[:, :, ::-1]
        sampledFrames.append(img)

    for idx in range(length):
        i = use_start_frm + idx * sampling_rate
        i = i % len(sampledFrames)
        video_output[idx] = sampledFrames[i]

    cap.release()
    return video_output

