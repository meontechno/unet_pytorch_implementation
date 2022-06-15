import os
from glob import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Class color encodings
LABEL_TO_COLOR = {0:[0,0,0], 1:[255,0,0], 2:[0,255,0]}


def mask2rgb(mask):
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)

    for i in np.unique(mask):
        rgb[mask == i] = LABEL_TO_COLOR[i]

    return rgb


def rgb2mask(rgb):
    mask = np.zeros((rgb.shape[0], rgb.shape[1]))

    for k, v in LABEL_TO_COLOR.items():
        mask[np.all(rgb == v, axis=2)] = k

    return mask


def test():
    # Load image and its corresponding mask
    img = Image.open('im1.jpg')
    mask = Image.open('im1_seg.png')

    # Configure plots
    fig, ax = plt.subplots(1, 2, figsize=(20, 20))

    # Plot original rgb image
    ax[0].imshow(img)

    # Optional conversion of rgb mask to flat mask
    # np_mask = np.array(mask)
    # mask = rgb2mask(np_mask)

    # Plot mask
    ax[1].imshow(mask)

    plt.show()


if __name__ == "__main__":
    test()
