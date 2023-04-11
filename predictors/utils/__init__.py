from . import torch_det_utils
from . import tf_det_utils
from . import license_utils
import cv2
import numpy as np
from typing import Tuple, Union


def letter_box(img: np.array,
               new_shape: Union[int, Tuple[int]] = (640, 640),
               auto: bool = False,
               scaleFill: bool = False,
               scaleup: bool = True,
               stride: int = 32):
    """
    img:  Array of image
    new_shape: the new_shape to resize to default (640,640)
    auto: converting the image to rectangle default False
    scaleFill: whether to stretch the image default False
    scaleup: whether to add the borders up or down default false
    stride: the stride of resizing
    return: Resized Array of image
    """
    shape = img.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))  # add border
    return img


def preprocess(img):
    y = img.copy()
    y = y.astype(np.float32) / 255
    return y[None]


def load_preprocess(img_path=None, img=None):
    if img_path is not None:
        img = cv2.imread(img_path)[..., ::-1]
    im = letter_box(img)
    im = preprocess(im)
    return im, img
