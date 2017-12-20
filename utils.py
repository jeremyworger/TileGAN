from __future__ import division
import math
import numpy as np 
import random
import json
import pprint
import scipy.misc
from time import gmtime, strftime
from six.moves import xrange

import tensorflow as tf 
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()
get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w * k_h * x.get_shape()[-1])

def get_image(img_path, height, width, resize_height=64, resize_width=64, 
                crop=True, grayscale=False):
    """Retrieves images from the dataset path.

    Args:
        img_path      (str): The image file path.
        height        (int): The current image height.
        width         (int): The current image width.
        resize_height (int): The new height after crop. Defaults to 64.
        resize_width  (int): The new width after crop. Defaults to 64.
        crop          (boolean): Performs a center crop. Defaults to true.
        grayscale     (boolean): Processes an image as a grayscale image.

    Returns:
            A transformed image.
    """
    image = imread(image_path, grayscale)
    return transform(image, height, width, 
                    resize_height, resize_width, crop)

def transform(image, height, width, resize_height=64, resize_width=64, crop=True):
    """Creates a new cropped image

    Args:
        image         (str): The image file path.
        height        (int): The current image height.
        width         (int): The current image width.
        resize_height (int): The new height after crop. Defaults to 64.
        resize_width  (int): The new width after crop. Defaults to 64.
        crop          (boolean): Performs a center crop. Defaults to true.

    Returns:
            A transformed image.
    """
    if crop:
        cropoed_image = center_crop(
            image, height, width, 
            reize_heigt, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropoed_image) / 127.5 - 1.

def center_crop(x, crop_height, crop_width, resize_height=64, resize_width=64):
    """Performs a center crop.

    Args:
        x               (int):  Image data
        crop_height     (int):  original height.
        crop_width      (int):  original width.
        resize_height   (int):  new image height. Defaults to 64 
        resize_width    (int):  new image width. Defaults to 64

    Returns:
            A new, resized image. 
    """
    if crop_width is None:
        crop_width = crop_height
    h, w = x.shape[:2]
    j = int(round((h - crop_height)/2.))
    i = int(round((w - crop_width)/2.))
    return scipy.misc.imresize(
        x[j:j+crop_height, i:i+crop_width], [resize_height, resize_width])