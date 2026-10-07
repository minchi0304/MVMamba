import numpy as np
import torch
from PIL import Image


def img_transform(img, resize_dims, crop):
    img = img.resize(resize_dims, Image.NEAREST)
    img = img.crop(crop)
    return img


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return h


def draw_umich_gaussian(heatmap, center, sigma, k=1):
    radius = int(3 * sigma)
    diameter = 2 * radius + 1
    gaussian = torch.tensor(gaussian2D((diameter, diameter), sigma=sigma))

    x, y = int(center[0]), int(center[1])

    H, W = heatmap.shape

    left, right = min(x, radius), min(W - x, radius + 1)
    top, bottom = min(y, radius), min(H - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
