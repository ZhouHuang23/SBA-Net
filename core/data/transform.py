# -*- coding: utf-8 -*-
from config.config import cfg

import random
random.seed(cfg.CUDNN.SEED)

import numpy as np
np.random.seed(cfg.CUDNN.SEED)

from PIL import Image, ImageFilter
from torchvision import transforms
import torchvision.transforms.functional as tf


def random_rotation(image, mask, gt, angle=None, p=1):
    if angle is None:
        angle = [-10, 10]
    if random.random() <= p:
        r = 0
        if isinstance(angle, list):
            r = random.randrange(angle[0], angle[1])
        else:
            assert "angle should be list type, please check the type..."
        image = image.rotate(r)
        mask = mask.rotate(r)
        gt = gt.rotate(r)

    return image, mask, gt


def random_flip(image, mask, gt, p=1):
    if random.random() <= p:
        if random.random() <= 0.5:
            image = tf.hflip(image)
            mask = tf.hflip(mask)
            gt = tf.hflip(gt)
        else:
            image = tf.vflip(image)
            mask = tf.vflip(mask)
            gt = tf.vflip(gt)

    return image, mask, gt


def random_resize(image, mask, gt, scale=[0.5, 2], p=1):
    if random.random() <= p:
        rows, cols = image.size[0], image.size[1]
        r = random.randint(scale[0] * 10, scale[1] * 10) / 10

        new_rows, new_cols = int(r * rows), int(r * cols)

        image = tf.resize(image, (new_rows, new_cols), Image.BILINEAR)  # image resize
        mask = tf.resize(mask, (new_rows, new_cols), Image.NEAREST)
        gt = tf.resize(gt, (new_rows, new_cols), Image.NEAREST)

        if new_rows > rows:
            image = tf.center_crop(image, output_size=(rows, cols))
            mask = tf.center_crop(mask, output_size=(rows, cols))
            gt = tf.center_crop(gt, output_size=(rows, cols))

        if new_rows < rows:
            padding = int((rows - new_rows) / 2)
            image = tf.pad(image, padding=padding, fill=0, padding_mode='constant')
            mask = tf.pad(mask, padding=padding, fill=0, padding_mode='constant')
            gt = tf.pad(gt, padding=padding, fill=0, padding_mode='constant')
            if padding * 2 + new_rows != rows:
                image = tf.resize(image, size=rows)
                mask = tf.resize(mask, size=rows)
                gt = tf.resize(gt, size=rows)

    return image, mask, gt


def adjust_contrast(image, mask, gt, scale=0.5, p=1):
    if random.random() <= p:
        image = tf.adjust_contrast(image, scale)
    return image, mask, gt


def adjust_brightness(image, mask, gt, factor=0.125, p=1):
    if random.random() <= p:
        image = tf.adjust_brightness(image, factor)
    return image, mask, gt


def adjust_saturation(image, mask, gt, factor=0.5, p=1):
    if random.random() <= p:
        image = tf.adjust_saturation(image, factor)
    return image, mask, gt


def adjust_hue(image, mask, gt, factor=0.2, p=1):
    if random.random() <= p:
        image = tf.adjust_hue(image, hue_factor=factor)
    return image, mask, gt


def center_crop(image, mask, gt, scale=1, p=1):
    if random.random() <= p:
        rows, cols = image.size[0], image.size[1]
        new_rows = int(rows * scale)
        new_cols = int(cols * scale)
        image = tf.center_crop(image, output_size=(new_rows, new_cols))
        mask = tf.center_crop(mask, output_size=(new_rows, new_cols))
        gt = tf.center_crop(gt, output_size=(new_rows, new_cols))

        new_rows, new_cols = image.size[0], image.size[1]

        padding = int((rows - new_rows) / 2)

        image = transforms.Pad(padding=padding, fill=0, padding_mode='constant')(image)
        mask = transforms.Pad(padding=padding, fill=0, padding_mode='constant')(mask)
        gt = transforms.Pad(padding=padding, fill=0, padding_mode='constant')(gt)

        if padding * 2 + new_rows != rows:
            image = tf.resize(image, size=rows)
            mask = tf.resize(mask, size=rows)
            gt = tf.resize(gt, size=rows)

    return image, mask, gt


def gaussian_blur(image, mask, gt, radius=3, p=1):
    if random.random() <= p:
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))

    return image, mask, gt


def add_gaussian_noise(image, mask, gt, noise_sigma=25, p=1):
    if random.random() <= p:
        temp_image = np.float64(np.copy(image))
        h, w, _ = temp_image.shape
        noise = np.random.randn(h, w) * noise_sigma
        noisy_image = np.zeros(temp_image.shape, np.float64)
        if len(temp_image.shape) == 2:
            image = temp_image + noise
        else:
            noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
            noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
            noisy_image[:, :, 2] = temp_image[:, :, 2] + noise

            image = Image.fromarray(np.uint8(noisy_image))

    return image, mask, gt


total_strategy = ['add_gaussian_noise', 'adjust_brightness', 'adjust_contrast', 'adjust_hue', 'adjust_saturation',
                  'center_crop', 'gaussian_blur', 'random_flip', 'random_resize', 'random_rotation']


def transform(aug_method, im, mask, gt):
    if aug_method == 'random_resize':
        im, mask, gt = random_resize(im, mask, gt)
    elif aug_method == 'random_rotation':
        im, mask, gt = random_rotation(im, mask, gt, angle=[-90, 90])
    elif aug_method == 'random_flip':
        im, mask, gt = random_flip(im, mask, gt)
    elif aug_method == 'gaussian_blur':
        im, mask, gt = gaussian_blur(im, mask, gt, radius=3)
    elif aug_method == 'gaussian_noise':
        im, mask, gt = add_gaussian_noise(im, mask, gt, noise_sigma=25)
    elif aug_method == 'adjust_brightness':
        im, mask, gt = adjust_brightness(im, mask, gt)
    elif aug_method == 'adjust_contrast':
        im, mask, gt = adjust_contrast(im, mask, gt)
    elif aug_method == 'adjust_hue':
        im, mask, gt = adjust_hue(im, mask, gt)
    elif aug_method == 'adjust_saturation':
        im, mask, gt = adjust_saturation(im, mask, gt)
    elif aug_method == 'center_crop':
        im, mask, gt = center_crop(im, mask, gt)
    return im, mask, gt


if __name__ == '__main__':
    z = 0

    # imsrc = r"\RS\ORSSD\train\image\0001.jpg"
    # gtsrc = r"\RS\ORSSD\train\mask\0001.png"
    # #
    # im = Image.open(imsrc).convert('RGB')
    # gt = Image.open(gtsrc).convert('L')
    # #
    # im, gt = random_rotation(im, gt)
    # #
    # from matplotlib import pyplot as plt
    # #
    # plt.subplot(121)
    # plt.imshow(np.asarray(im))
    # plt.subplot(122)
    # plt.imshow(np.asarray(gt))
    # plt.show()
