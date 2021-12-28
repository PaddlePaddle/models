import numbers
import warnings
from enum import Enum

import numpy as np

import paddle
from paddle import Tensor
from typing import List, Tuple, Any, Optional

try:
    import accimage
except ImportError:
    accimage = None

from . import functional_pil as F_pil
from . import functional_tensor as F_t


class InterpolationMode(Enum):
    """Interpolation modes
    Available interpolation methods are ``nearest``, ``bilinear``, ``bicubic``, ``box``, ``hamming``, and ``lanczos``.
    """
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


def _interpolation_modes_from_int(i: int) -> InterpolationMode:
    inverse_modes_mapping = {
        0: InterpolationMode.NEAREST,
        2: InterpolationMode.BILINEAR,
        3: InterpolationMode.BICUBIC,
        4: InterpolationMode.BOX,
        5: InterpolationMode.HAMMING,
        1: InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[i]


pil_modes_mapping = {
    InterpolationMode.NEAREST: 0,
    InterpolationMode.BILINEAR: 2,
    InterpolationMode.BICUBIC: 3,
    InterpolationMode.BOX: 4,
    InterpolationMode.HAMMING: 5,
    InterpolationMode.LANCZOS: 1,
}


def _is_numpy(img: Any) -> bool:
    return isinstance(img, np.ndarray)


def _is_numpy_image(img: Any) -> bool:
    return img.ndim in {2, 3}


def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See :class:`~paddlevision.transforms.ToTensor` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not (F_pil._is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(
            type(pic)))

    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.
                         format(pic.ndim))

    default_float_dtype = paddle.get_default_dtype()

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]
        img = paddle.to_tensor(pic.transpose((2, 0, 1)))
        # backward compatibility
        if not img.dtype == default_float_dtype:
            img = img.astype(dtype=default_float_dtype)
            return img.divide(paddle.full_like(img, 255))
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros(
            [pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return paddle.to_tensor(nppic).astype(dtype=default_float_dtype)

    # handle PIL Image
    mode_to_nptype = {'I': np.int32, 'I;16': np.int16, 'F': np.float32}
    img = paddle.to_tensor(
        np.array(
            pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))

    if pic.mode == '1':
        img = 255 * img
    img = img.reshape([pic.size[1], pic.size[0], len(pic.getbands())])

    if not img.dtype == default_float_dtype:
        img = img.astype(dtype=default_float_dtype)
        # put it from HWC to CHW format
        img = img.transpose((2, 0, 1))
        return img.divide(paddle.full_like(img, 255))
    else:
        # put it from HWC to CHW format
        img = img.transpose((2, 0, 1))
        return img


def normalize(tensor: Tensor,
              mean: List[float],
              std: List[float],
              inplace: bool=False) -> Tensor:
    """Normalize a float tensor image with mean and standard deviation.
    This transform does not support PIL Image.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~paddlevision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Float tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not isinstance(tensor, paddle.Tensor):
        raise TypeError('Input tensor should be a paddle tensor. Got {}.'.
                        format(type(tensor)))

    if not tensor.dtype in (paddle.float16, paddle.float32, paddle.float64):
        raise TypeError('Input tensor should be a float tensor. Got {}.'.
                        format(tensor.dtype))

    if tensor.ndim < 3:
        raise ValueError(
            'Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.shape() = '
            '{}.'.format(tensor.shape))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = paddle.to_tensor(mean, dtype=dtype, place=tensor.place)
    std = paddle.to_tensor(std, dtype=dtype, place=tensor.place)
    if (std == 0).any():
        raise ValueError('std evaluated to zero, leading to division by zero.')
    if mean.ndim == 1:
        mean = mean.reshape((-1, 1, 1))
    if std.ndim == 1:
        std = std.reshape((-1, 1, 1))
    tensor = tensor.subtract(mean).divide(std)
    return tensor


def resize(img: Tensor,
           size: List[int],
           interpolation: InterpolationMode=InterpolationMode.BILINEAR,
           max_size: Optional[int]=None,
           antialias: Optional[bool]=None) -> Tensor:
    r"""Resize the input image to the given size.
    If the image is paddle Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. warning::
        The output image might be different depending on its type: when downsampling, the interpolation of PIL images
        and tensors is slightly different, because PIL applies antialiasing. This may lead to significant differences
        in the performance of a network. Therefore, it is preferable to train and serve a model with the same input
        types. See also below the ``antialias`` parameter, which can help making the output of PIL images and tensors
        closer.

    Args:
        img (PIL Image or Tensor): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`.

        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`paddlevision.transforms.InterpolationMode`.
            Default is ``InterpolationMode.BILINEAR``. If input is Tensor, only ``InterpolationMode.NEAREST``,
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        max_size (int, optional): The maximum allowed for the longer edge of
            the resized image: if the longer edge of the image is greater
            than ``max_size`` after being resized according to ``size``, then
            the image is resized again so that the longer edge is equal to
            ``max_size``. As a result, ``size`` might be overruled, i.e the
            smaller edge may be shorter than ``size``.
        antialias (bool, optional): antialias flag. If ``img`` is PIL Image, the flag is ignored and anti-alias
            is always used. If ``img`` is Tensor, the flag is False by default and can be set to True for
            ``InterpolationMode.BILINEAR`` only mode. This can help making the output for PIL images and tensors
            closer.

            .. warning::
                There is no autodiff support for ``antialias=True`` option with input ``img`` as Tensor.

    Returns:
        PIL Image or Tensor: Resized image.
    """
    # Backward compatibility with integer value
    if isinstance(interpolation, int):
        warnings.warn(
            "Argument interpolation should be of type InterpolationMode instead of int. "
            "Please, use InterpolationMode enum.")
        interpolation = _interpolation_modes_from_int(interpolation)

    if not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode")

    if not isinstance(img, paddle.Tensor):
        if antialias is not None and not antialias:
            warnings.warn(
                "Anti-alias option is always applied for PIL Image input. Argument antialias is ignored."
            )
        pil_interpolation = pil_modes_mapping[interpolation]
        return F_pil.resize(
            img, size=size, interpolation=pil_interpolation, max_size=max_size)

    return F_t.resize(
        img,
        size=size,
        interpolation=interpolation.value,
        max_size=max_size,
        antialias=antialias)


def _get_image_size(img: Tensor) -> List[int]:
    """Returns image size as [w, h]
    """
    if isinstance(img, paddle.Tensor):
        return F_t._get_image_size(img)

    return F_pil._get_image_size(img)


def pad(img: Tensor,
        padding: List[int],
        fill: int=0,
        padding_mode: str="constant") -> Tensor:
    r"""Pad the given image on all sides with the given "pad" value.
    If the image is paddle Tensor, it is expected
    to have [..., H, W] shape, where ... means at most 2 leading dimensions for mode reflect and symmetric,
    at most 3 leading dimensions for mode edge,
    and an arbitrary number of leading dimensions for mode constant

    Args:
        img (PIL Image or Tensor): Image to be padded.
        padding (int or sequence): Padding on each border. If a single int is provided this
            is used to pad all borders. If sequence of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.
        fill (number or str or tuple): Pixel fill value for constant fill. Default is 0.
            If a tuple of length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
            Only number is supported for paddle Tensor.
            Only int or str or tuple value is supported for PIL Image.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value at the edge of the image.
              If input a 5D paddle Tensor, the last 3 dimensions will be padded instead of the last 2

            - reflect: pads with reflection of image without repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
              will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
              will result in [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        PIL Image or Tensor: Padded image.
    """
    if not isinstance(img, paddle.Tensor):
        return F_pil.pad(img,
                         padding=padding,
                         fill=fill,
                         padding_mode=padding_mode)

    return F_t.pad(img, padding=padding, fill=fill, padding_mode=padding_mode)


def crop(img: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    """Crop the given image at specified location and output size.
    If the image is paddle Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        PIL Image or Tensor: Cropped image.
    """

    if not isinstance(img, paddle.Tensor):
        return F_pil.crop(img, top, left, height, width)

    return F_t.crop(img, top, left, height, width)


def center_crop(img: Tensor, output_size: List[int]) -> Tensor:
    """Crops the given image at the center.
    If the image is paddle Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    image_width, image_height = _get_image_size(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2
            if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2
            if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2
            if crop_height > image_height else 0,
        ]
        img = pad(img, padding_ltrb, fill=0)  # PIL uses fill value 0
        image_width, image_height = _get_image_size(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return crop(img, crop_top, crop_left, crop_height, crop_width)


def resized_crop(
        img: Tensor,
        top: int,
        left: int,
        height: int,
        width: int,
        size: List[int],
        interpolation: InterpolationMode=InterpolationMode.BILINEAR) -> Tensor:
    """Crop the given image and resize it to desired size.
    If the image is paddle Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        img (PIL Image or Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        size (sequence or int): Desired output size. Same semantics as ``resize``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`paddlevision.transforms.InterpolationMode`.
            Default is ``InterpolationMode.BILINEAR``. If input is Tensor, only ``InterpolationMode.NEAREST``,
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation)
    return img
