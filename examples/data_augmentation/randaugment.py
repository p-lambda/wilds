# Adapted from https://github.com/kekmodel/FixMatch-pytorch

import numpy as np
import torch
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

_PARAMETER_MAX = 10


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = _sample_uniform(a=0, b=w)
    y0 = _sample_uniform(a=0, b=h)
    x0 = int(max(0, x0 - v / 2.0))
    y0 = int(max(0, y0 - v / 2.0))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if _sample_uniform(a=0, b=1) < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if _sample_uniform(a=0, b=1) < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if _sample_uniform(a=0, b=1) < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if _sample_uniform(a=0, b=1) < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if _sample_uniform(a=0, b=1) < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if _sample_uniform(a=0, b=1) < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _sample_uniform(a, b):
    return torch.FloatTensor(1).uniform_(a, b).item()


def _float_parameter(v, max_v):
    return float(v) * max_v / _PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / _PARAMETER_MAX)


class RandAugment(object):
    def __init__(self, n, m, augmentation_pool):
        assert n >= 1, "RandAugment N has to be a value greater than 1."
        assert (
            m >= 1 and m <= 10
        ), "RandAugment M has to be a value between 1 and 10 inclusive."
        self.n = n
        self.m = m
        self.augmentation_pool = augmentation_pool

    def __call__(self, img):
        # Choose n augmentations with replacement. Equivalent to random.choices(self.augmentation_pool, k=self.n).
        ops = [
            self.augmentation_pool[torch.randint(len(self.augmentation_pool), (1,))]
            for _ in range(self.n)
        ]
        for op, max_v, bias in ops:
            v = torch.randint(low=1, high=self.m, size=(1,)).item()
            if _sample_uniform(a=0, b=1) < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32 * 0.5))
        return img


# Pool of augmentations used in the original FixMatch paper:
# https://github.com/google-research/fixmatch/blob/d4985a158065947dba803e626ee9a6721709c570/libml/augment.py#L37
FIX_MATCH_AUGMENTATION_POOL = [
    (AutoContrast, None, None),
    (Brightness, 0.9, 0.05),
    (Color, 0.9, 0.05),
    (Contrast, 0.9, 0.05),
    (Equalize, None, None),
    (Identity, None, None),
    (Posterize, 4, 4),
    (Rotate, 30, 0),
    (Sharpness, 0.9, 0.05),
    (ShearX, 0.3, 0),
    (ShearY, 0.3, 0),
    (Solarize, 256, 0),
    (TranslateX, 0.3, 0),
    (TranslateY, 0.3, 0),
]
