# -*- coding: utf-8 -*-
import os
import scipy.io as io
import cv2
from core.libs import imutils
import torch
import imageio

from config.config import cfg
from core.libs.logger import set_logger

try:
    from . import transform4cv2_fetch2 as tf
    from . import transform4cv2_fetch1 as tf1
except:
    from torchvision.transforms import transforms as tf

from torchvision.transforms import transforms as tf2

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random

random.seed(cfg.CUDNN.SEED)

import math
from core.data import transform
import numpy as np

np.random.seed(cfg.CUDNN.SEED)

logger = set_logger()


def _make_dataset(im_dir, scr_dir):
    imgs_image = []
    imgs_scr = []

    for f in os.listdir(im_dir):
        if f.endswith(cfg.DATASET.FORMAT_TRAIN_SET):
            imgs_image.append(os.path.join(im_dir, f))
    for f in os.listdir(scr_dir):
        if f.endswith(cfg.DATASET.FORMAT_MASK):
            imgs_scr.append(os.path.join(scr_dir, f))

    return imgs_image, imgs_scr


class MyDataSet(Dataset):
    def __init__(self, dataset_dir, aug_is_need=True, p=None):
        im_dir = os.path.join(dataset_dir, "image")
        scr_dir = os.path.join(dataset_dir, "scribble")

        mean_std = io.loadmat(cfg.DATASET.MEAN_STD)
        mean = mean_std['mean'][0]
        std = mean_std['std'][0]

        self.train_img_size = cfg.TRAIN.TRAIN_IMG_SIZE

        self.im_scr_transform = tf.Compose(
            tf.Normalize(mean=mean, std=std),
            tf.Resize(self.train_img_size, self.train_img_size),
            tf.RandomHorizontalFlip(),
            # tf.RandomCrop(288, 288),
            tf.ToTensor()
        )

        self.im_list, self.scr_list = _make_dataset(im_dir=im_dir, scr_dir=scr_dir)

    def __getitem__(self, index):
        # load image
        im_filepath = self.im_list[index]
        scr_filepath = self.scr_list[index]

        im = cv2.imread(im_filepath).astype(np.float32)[:, :, ::-1]
        scr = cv2.imread(scr_filepath).astype(np.float32)[:, :, ::-1]

        if im is None or scr is None:
            print(im_filepath)

        im, scr = self.im_scr_transform(im, scr)

        mask = scr.clone()
        gt = mask.clone()

        mask[mask == 2.0] = 255.0
        mask[mask == 1.0] = 255.0
        mask = mask / 255

        gt[gt == 2.0] = 0.
        gt[gt == 1.0] = 255.0
        gt = gt / 255

        return im, mask, gt

    def __len__(self):
        return len(self.im_list)


def load_train_data():
    dataset = MyDataSet(dataset_dir=cfg.DATASET.TRAIN_SET, p=cfg.TRAIN.DEFAULT_AUG_P)
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                            num_workers=cfg.DATALOADER.NUM_WORKERS, drop_last=True, pin_memory=True)
    dt_size = math.floor(len(dataloader.dataset) / cfg.TRAIN.BATCH_SIZE)

    logger.warning('Training set: total {} images in dir: {}'.format(dataset.__len__(), cfg.DATASET.TRAIN_SET))
    return dataloader, dt_size


def load_val_data():
    dataset = MyDataSet(dataset_dir=cfg.DATASET.VAL_SET, aug_is_need=False)
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                            num_workers=cfg.DATALOADER.NUM_WORKERS, drop_last=True, pin_memory=True)
    dt_size = math.floor(len(dataloader.dataset) / cfg.TRAIN.BATCH_SIZE)
    logger.warning('Validation set: total {} images in dir: {}'.format(dataset.__len__(), cfg.DATASET.VAL_SET))
    return dataloader, dt_size


class MyTestDataSet(Dataset):
    def __init__(self, dataset_dir, test_size=352):
        im_dir = os.path.join(dataset_dir, 'image')
        gt_dir = os.path.join(dataset_dir, 'mask')

        mean_std = io.loadmat(cfg.DATASET.MEAN_STD)
        mean = mean_std['mean'][0]
        std = mean_std['std'][0]

        self.transform = tf.Compose(
            tf.Normalize(mean=mean, std=std),
            tf.Resize(test_size, test_size),
            tf.ToTensor()
        )

        self.im_list, self.gt_list = _make_dataset(im_dir=im_dir, scr_dir=gt_dir)

    def __getitem__(self, index):
        im_path = self.im_list[index]
        gt_path = self.gt_list[index]

        image = cv2.imread(im_path).astype(np.float32)[:, :, ::-1]
        gt = cv2.imread(gt_path).astype(np.float32)[:, :, ::-1]
        H, W, C = gt.shape

        image, gt = self.transform(image, gt)

        name = self.im_list[index].split('\\')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        return image, gt, (H, W), name, im_path

    def __len__(self):
        return len(self.im_list)


def load_test_data():
    dataset = MyTestDataSet(dataset_dir=cfg.DATASET.TEST_SET)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)
    logger.warning('Test set: total {} images in dir: {}'.format(dataset.__len__(), cfg.DATASET.TEST_SET))

    return dataloader


def _load_image_label_list_from_npy(npy_path):
    return np.load(npy_path, allow_pickle=True).astype(np.float32)


class MyTrainCamDataSet(Dataset):
    def __init__(self, dataset_dir, aug_is_need=True, p=None):
        im_dir = os.path.join(dataset_dir, "image")
        scr_dir = os.path.join(dataset_dir, "scribble")

        mean_std = io.loadmat(cfg.DATASET.MEAN_STD)
        mean = mean_std['mean'][0]
        std = mean_std['std'][0]

        self.train_img_size = cfg.TRAIN.TRAIN_IMG_SIZE

        self.im_transform = tf1.Compose(
            tf1.Normalize(mean=mean, std=std),
            tf1.Resize(self.train_img_size, self.train_img_size),
            tf1.RandomHorizontalFlip(),
            # tf.RandomCrop(288, 288),
            tf1.ToTensor()
        )

        self.im_list, _ = _make_dataset(im_dir=im_dir, scr_dir=scr_dir)
        self.label_list = _load_image_label_list_from_npy(dataset_dir + r'\cls_labels.npy')

    def __getitem__(self, index):
        # load image
        im_filepath = self.im_list[index]

        img = np.asarray(imageio.imread(im_filepath))

        if img is None:
            print(im_filepath)

        img = self.im_transform(img)

        name = self.im_list[index].split('\\')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0]

        label = torch.from_numpy(self.label_list[index])

        return name, img, label

    def __len__(self):
        return len(self.im_list)


def load_train_CAM_data():
    dataset = MyTrainCamDataSet(dataset_dir=cfg.DATASET.TRAIN_SET, p=cfg.TRAIN.DEFAULT_AUG_P)
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                            num_workers=cfg.DATALOADER.NUM_WORKERS, drop_last=True, pin_memory=True)
    dt_size = math.floor(len(dataloader.dataset) / cfg.TRAIN.BATCH_SIZE)

    logger.warning('Training set: total {} images in dir: {}'.format(dataset.__len__(), cfg.DATASET.TRAIN_SET))
    return dataloader, dt_size


class MyMakeCamDataSet(Dataset):
    def __init__(self, dataset_dir, aug_is_need=True, p=None, scales=(1.0,)):
        self.scales = scales
        im_dir = os.path.join(dataset_dir, "image")
        scr_dir = os.path.join(dataset_dir, "scribble")

        mean_std = io.loadmat(cfg.DATASET.MEAN_STD)
        mean = mean_std['mean'][0]
        std = mean_std['std'][0]

        self.train_img_size = cfg.TRAIN.TRAIN_IMG_SIZE

        self.im_transform = tf1.Compose(
            tf1.Normalize(mean=mean, std=std)
        )

        self.im_list, _ = _make_dataset(im_dir=im_dir, scr_dir=scr_dir)

    def __getitem__(self, index):
        # load image
        im_filepath = self.im_list[index]
        # img = cv2.imread(im_filepath).astype(np.float32)[:, :, ::-1]
        img = imageio.imread(im_filepath)

        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)

            s_img = self.im_transform(s_img)

            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
            # ms_img_list.append(s_img)
        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]

        name = self.im_list[index].split('\\')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0]

        label = torch.ones(1)

        return name, ms_img_list, (img.shape[0], img.shape[1]), label

    def __len__(self):
        return len(self.im_list)


def make_CAM_dataset():
    dataset = MyMakeCamDataSet(dataset_dir=cfg.DATASET.TRAIN_SET, p=cfg.TRAIN.DEFAULT_AUG_P)
    return dataset


class MyMakeBoundaryLabelDataSet(Dataset):
    def __init__(self, dataset_dir):
        im_dir = os.path.join(dataset_dir, "image")
        scr_dir = os.path.join(dataset_dir, "scribble")

        mean_std = io.loadmat(cfg.DATASET.MEAN_STD)
        mean = mean_std['mean'][0]
        std = mean_std['std'][0]

        self.train_img_size = cfg.TRAIN.TRAIN_IMG_SIZE

        self.im_transform = tf1.Compose(
            tf1.Normalize(mean=mean, std=std)
        )

        self.im_list, _ = _make_dataset(im_dir=im_dir, scr_dir=scr_dir)

    def __getitem__(self, index):
        # load image
        im_filepath = self.im_list[index]

        img = np.asarray(imageio.imread(im_filepath))

        name = self.im_list[index].split('\\')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0]

        return name, img

    def __len__(self):
        return len(self.im_list)


def make_boundary_label_dataset():
    dataset = MyMakeBoundaryLabelDataSet(dataset_dir=cfg.DATASET.TRAIN_SET)
    return dataset
