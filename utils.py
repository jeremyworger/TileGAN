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

def imread(path, grayscale=False):
    """Reads an image
    Args:
        path        (str):  Path to image.
        grayscale   (boolean):  Read an image without color.
    """
    if grayscale:
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def save_images(images, size, img_path):
    return imsave(inverse_transform(images), size, img_path)

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

def inverse_transform(images):
    return (images+1.)/2.

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


def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3, 4):
        c = images.shape[3]
        img = np.zeroes((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    elif images.shape[3] == 1:
        img = np.zeroes((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('Images parameters must have dimensions HxW, HxWx3, HxWx4')

def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy 

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]
        
        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1) / 2*255).astype(np.uint8)
    
    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps = len(images) / duration)


def to_json(output_path, *layers):
  with open(output_path, "w") as layer_f:
    lines = ""
    for w, b, bn in layers:
      layer_idx = w.name.split('/')[0].split('h')[1]

      B = b.eval()

      if "lin/" in w.name:
        W = w.eval()
        depth = W.shape[1]
      else:
        W = np.rollaxis(w.eval(), 2, 0)
        depth = W.shape[0]

      biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
      if bn != None:
        gamma = bn.gamma.eval()
        beta = bn.beta.eval()

        gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
        beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
      else:
        gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
        beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

      if "lin/" in w.name:
        fs = []
        for w in W.T:
          fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

        lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
      else:
        fs = []
        for w_ in W:
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

        lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))
