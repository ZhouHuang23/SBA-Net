#!/usr/bin/python3
#coding=utf-8

import cv2
import torch
import numpy as np

class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, scribble):
        for op in self.ops:
            image, scribble = op(image, scribble)
        return image, scribble

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, image, scribble):
        image = (image - self.mean)/self.std
        # scribble /= 255
        return image, scribble

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, scribble):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        scribble  = cv2.resize( scribble, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, scribble

class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, scribble):
        H,W,_ = image.shape
        xmin  = np.random.randint(W-self.W+1)
        ymin  = np.random.randint(H-self.H+1)
        image = image[ymin:ymin+self.H, xmin:xmin+self.W, :]
        scribble  = scribble[ymin:ymin+self.H, xmin:xmin+self.W, :]
        return image, scribble

class RandomHorizontalFlip(object):
    def __call__(self, image, scribble):
        if np.random.randint(2)==1:
            image = image[:,::-1,:].copy()
            scribble  =  scribble[:,::-1,:].copy()
        return image, scribble

class ToTensor(object):
    def __call__(self, image, scribble):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        scribble  = torch.from_numpy(scribble)
        scribble  = scribble.permute(2, 0, 1)
        return image, scribble.mean(dim=0, keepdim=True)

