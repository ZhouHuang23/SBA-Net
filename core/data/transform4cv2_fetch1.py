#!/usr/bin/python3
#coding=utf-8

import cv2
import torch
import numpy as np

class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image):
        for op in self.ops:
            image = op(image)
        return image

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, image):
        image = (image - self.mean)/self.std

        return image

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)

        return image

class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image):
        H,W,_ = image.shape
        xmin  = np.random.randint(W-self.W+1)
        ymin  = np.random.randint(H-self.H+1)
        image = image[ymin:ymin+self.H, xmin:xmin+self.W, :]
        return image

class RandomHorizontalFlip(object):
    def __call__(self, image):
        if np.random.randint(2)==1:
            image = image[:,::-1,:].copy()

        return image

class ToTensor(object):
    def __call__(self, image):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)

        return image

