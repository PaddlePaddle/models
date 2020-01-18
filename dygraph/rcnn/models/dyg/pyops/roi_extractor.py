import sys
import math
import numpy as np
from paddle.fluid.dygraph.base import to_variable


def roi_pool(input_x, rois, pooled_height, pooled_width, spatial_scale):
    input_x = input_x.numpy()
    rois = rois.numpy()
    batch_size, channels, height, width = input_x.shape
    print("debug roi pool")
    print("debug input feat: ", input_x.shape)
    rois_num = rois.shape[1]
    #out_data = np.zeros((rois_num, channels, pooled_height, pooled_width))
    #argmax_data = np.zeros((rois_num, channels, pooled_height, pooled_width))
    outs_list = []
    for bi in range(batch_size):
        out_data = np.zeros((rois_num, channels, pooled_height, pooled_width))
        argmax_data = np.zeros(
            (rois_num, channels, pooled_height, pooled_width))
        for i in range(rois_num):
            roi = rois[bi][i]
            # roi_batch_id = int(roi[0])
            roi_start_w = int(np.round(roi[0] * spatial_scale))
            roi_start_h = int(np.round(roi[1] * spatial_scale))
            roi_end_w = int(np.round(roi[2] * spatial_scale))
            roi_end_h = int(np.round(roi[3] * spatial_scale))

            roi_height = int(max(roi_end_h - roi_start_h + 1, 1))
            roi_width = int(max(roi_end_w - roi_start_w + 1, 1))

            x_i = input_x[bi]  #input_x[roi_batch_id]

            bin_size_h = float(roi_height) / float(pooled_height)
            bin_size_w = float(roi_width) / float(pooled_width)

            for c in range(channels):
                for ph in range(pooled_height):
                    for pw in range(pooled_width):
                        hstart = int(math.floor(ph * bin_size_h))
                        wstart = int(math.floor(pw * bin_size_w))
                        hend = int(math.ceil((ph + 1) * bin_size_h))
                        wend = int(math.ceil((pw + 1) * bin_size_w))

                        hstart = min(max(hstart + roi_start_h, 0), height)
                        hend = min(max(hend + roi_start_h, 0), height)
                        wstart = min(max(wstart + roi_start_w, 0), width)
                        wend = min(max(wend + roi_start_w, 0), width)

                        is_empty = (hend <= hstart) or (wend <= wstart)
                        if is_empty:
                            out_data[i, c, ph, pw] = 0
                        else:
                            out_data[i, c, ph, pw] = -sys.float_info.max

                        argmax_data[i, c, ph, pw] = -1

                        for h in range(hstart, hend):
                            for w in range(wstart, wend):
                                if x_i[c, h, w] > out_data[i, c, ph, pw]:
                                    out_data[i, c, ph, pw] = x_i[c, h, w]
                                    argmax_data[i, c, ph, pw] = h * width + w

        outs = out_data.astype('float32')
        argmaxes = argmax_data.astype('int64')
        outs_list.append(outs)
    outs = np.asarray(outs_list, dtype=np.float32)
    outs = to_variable(np.asarray(outs_list, dtype=np.float32))
    return outs
