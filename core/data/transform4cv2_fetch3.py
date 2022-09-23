#!/usr/bin/python3
# coding=utf-8

import cv2
import torch
import numpy as np


class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, scribble, edge):
        for op in self.ops:
            image, scribble,edge= op(image, scribble, edge)
        return image, scribble, edge


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, scribble, edge):
        image = (image - self.mean) / self.std
        return image, scribble, edge


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, scribble, edge):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        scribble = cv2.resize(scribble, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        edge = cv2.resize(edge, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, scribble, edge


class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, scribble, edge):
        H, W, _ = image.shape
        xmin = np.random.randint(W - self.W + 1)
        ymin = np.random.randint(H - self.H + 1)
        image = image[ymin:ymin + self.H, xmin:xmin + self.W, :]
        scribble = scribble[ymin:ymin + self.H, xmin:xmin + self.W, :]
        edge = edge[ymin:ymin + self.H, xmin:xmin + self.W, :]
        return image, scribble, edge


class RandomHorizontalFlip(object):
    def __call__(self, image, scribble, edge):
        if np.random.randint(2) == 1:
            image = image[:, ::-1, :].copy()
            scribble = scribble[:, ::-1, :].copy()
            edge = edge[:, ::-1, :].copy()
        return image, scribble, edge


class ToTensor(object):
    def __call__(self, image, scribble, edge):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        scribble = torch.from_numpy(scribble)
        scribble = scribble.permute(2, 0, 1)
        edge = torch.from_numpy(edge)
        edge = edge.permute(2, 0, 1)
        return image, scribble.mean(dim=0, keepdim=True), edge.mean(dim=0, keepdim=True)
