import warnings

import paddle
from paddle import Tensor
from paddle.nn.functional import grid_sample, conv2d, interpolate, pad as paddle_pad
from typing import Optional, Tuple, List


def _is_tensor_a_paddle_image(x: Tensor) -> bool:
    return x.ndim >= 2


def _assert_image_tensor(img):
    if not _is_tensor_a_paddle_image(img):
        raise TypeError("Tensor is not a paddle image.")


def _get_image_size(img: Tensor) -> List[int]:
    # Returns (w, h) of tensor image
    _assert_image_tensor(img)
    return [img.shape[-1], img.shape[-2]]


def _cast_squeeze_in(img: Tensor, req_dtypes: List[paddle.dtype]) -> Tuple[
        Tensor, bool, bool, paddle.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.as_type(req_dtype)
    return img, need_cast, need_squeeze, out_dtype


def _cast_squeeze_out(img: Tensor,
                      need_cast: bool,
                      need_squeeze: bool,
                      out_dtype: paddle.dtype):
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (paddle.uint8, paddle.int8, paddle.int16, paddle.int32,
                         paddle.int64):
            # it is better to round before cast
            img = paddle.round(img)
        img = img.as_type(out_dtype)

    return img


def _pad_symmetric(img: Tensor, padding: List[int]) -> Tensor:
    # padding is left, right, top, bottom

    # crop if needed
    if padding[0] < 0 or padding[1] < 0 or padding[2] < 0 or padding[3] < 0:
        crop_left, crop_right, crop_top, crop_bottom = [
            -min(x, 0) for x in padding
        ]
        img = img[..., crop_top:img.shape[-2] - crop_bottom, crop_left:
                  img.shape[-1] - crop_right]
        padding = [max(x, 0) for x in padding]

    in_sizes = img.size()

    x_indices = [i for i in range(in_sizes[-1])]  # [0, 1, 2, 3, ...]
    left_indices = [i for i in
                    range(padding[0] - 1, -1, -1)]  # e.g. [3, 2, 1, 0]
    right_indices = [-(i + 1) for i in range(padding[1])]  # e.g. [-1, -2, -3]
    x_indices = paddle.to_tensor(
        left_indices + x_indices + right_indices, device=img.device)

    y_indices = [i for i in range(in_sizes[-2])]
    top_indices = [i for i in range(padding[2] - 1, -1, -1)]
    bottom_indices = [-(i + 1) for i in range(padding[3])]
    y_indices = paddle.to_tensor(
        top_indices + y_indices + bottom_indices, device=img.device)

    ndim = img.ndim
    if ndim == 3:
        return img[:, y_indices[:, None], x_indices[None, :]]
    elif ndim == 4:
        return img[:, :, y_indices[:, None], x_indices[None, :]]
    else:
        raise RuntimeError(
            "Symmetric padding of N-D tensors are not supported yet")


def crop(img: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    _assert_image_tensor(img)

    w, h = _get_image_size(img)
    right = left + width
    bottom = top + height

    if left < 0 or top < 0 or right > w or bottom > h:
        padding_ltrb = [
            max(-left, 0), max(-top, 0), max(right - w, 0), max(bottom - h, 0)
        ]
        return pad(img[..., max(top, 0):bottom, max(left, 0):right],
                   padding_ltrb,
                   fill=0)
    return img[..., top:bottom, left:right]


def pad(img: Tensor,
        padding: List[int],
        fill: int=0,
        padding_mode: str="constant") -> Tensor:
    _assert_image_tensor(img)

    if not isinstance(padding, (int, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill, (int, float)):
        raise TypeError("Got inappropriate fill arg")
    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, tuple):
        padding = list(padding)

    if isinstance(padding, list) and len(padding) not in [1, 2, 4]:
        raise ValueError(
            "Padding must be an int or a 1, 2, or 4 element tuple, not a " +
            "{} element tuple".format(len(padding)))

    if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
        raise ValueError(
            "Padding mode should be either constant, edge, reflect or symmetric"
        )

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif len(padding) == 1:
        pad_left = pad_right = pad_top = pad_bottom = padding[0]
    elif len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    else:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    p = [pad_left, pad_right, pad_top, pad_bottom]

    if padding_mode == "edge":
        # remap padding_mode str
        padding_mode = "replicate"
    elif padding_mode == "symmetric":
        # route to another implementation
        return _pad_symmetric(img, p)

    need_squeeze = False
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if (padding_mode != "constant") and img.dtype not in (paddle.float32,
                                                          paddle.float64):
        # Here we temporary cast input tensor to float
        need_cast = True
        img = img.as_type(paddle.float32)

    img = paddle_pad(img, p, mode=padding_mode, value=float(fill))

    if need_squeeze:
        img = img.squeeze(axis=0)

    if need_cast:
        img = img.as_type(out_dtype)

    return img


def resize(img: Tensor,
           size: List[int],
           interpolation: str="bilinear",
           max_size: Optional[int]=None,
           antialias: Optional[bool]=None) -> Tensor:
    _assert_image_tensor(img)

    if not isinstance(size, (int, tuple, list)):
        raise TypeError("Got inappropriate size arg")
    if not isinstance(interpolation, str):
        raise TypeError("Got inappropriate interpolation arg")

    if interpolation not in ["nearest", "bilinear", "bicubic"]:
        raise ValueError(
            "This interpolation mode is unsupported with Tensor input")

    if isinstance(size, tuple):
        size = list(size)

    if isinstance(size, list):
        if len(size) not in [1, 2]:
            raise ValueError(
                "Size must be an int or a 1 or 2 element tuple/list, not a "
                "{} element tuple/list".format(len(size)))
        if max_size is not None and len(size) != 1:
            raise ValueError(
                "max_size should only be passed if size specifies the length of the smaller edge."
            )

    if antialias is None:
        antialias = False

    if antialias and interpolation not in ["bilinear", "bicubic"]:
        raise ValueError(
            "Antialias option is supported for bilinear and bicubic interpolation modes only"
        )

    w, h = _get_image_size(img)

    if isinstance(size, int) or len(
            size) == 1:  # specified size only for the smallest edge
        short, long = (w, h) if w <= h else (h, w)
        requested_new_short = size if isinstance(size, int) else size[0]

        if short == requested_new_short:
            return img

        new_short, new_long = requested_new_short, int(requested_new_short *
                                                       long / short)

        if max_size is not None:
            if max_size <= requested_new_short:
                raise ValueError(
                    f"max_size = {max_size} must be strictly greater than the requested "
                    f"size for the smaller edge size = {size}")
            if new_long > max_size:
                new_short, new_long = int(max_size * new_short /
                                          new_long), max_size

        new_w, new_h = (new_short, new_long) if w <= h else (new_long,
                                                             new_short)

    else:  # specified both h and w
        new_w, new_h = size[1], size[0]

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(
        img, [paddle.float32, paddle.float64])

    # Define align_corners to avoid warnings
    align_corners = False if interpolation in ["bilinear", "bicubic"] else None

    img = interpolate(
        img,
        size=[new_h, new_w],
        mode=interpolation,
        align_corners=align_corners)

    if interpolation == "bicubic" and out_dtype == paddle.uint8:
        img = img.clamp(min=0, max=255)

    img = _cast_squeeze_out(
        img,
        need_cast=need_cast,
        need_squeeze=need_squeeze,
        out_dtype=out_dtype)

    return img
