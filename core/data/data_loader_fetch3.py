# -*- coding: utf-8 -*-
import os
import scipy.io as io
import cv2

from . import transform4cv2_fetch3 as mytf_fetch3
from . import transform4cv2_fetch1 as mytf_fetch1
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
from config.config import cfg
from core.libs.logger import set_logger

random.seed(cfg.CUDNN.SEED)

import math
import numpy as np

np.random.seed(cfg.CUDNN.SEED)

logger = set_logger()


def _make_dataset(*dir):
    if len(dir) == 1:
        im_dir = str(dir[0])
        imgs_image = []
        for f in os.listdir(im_dir):
            if f.endswith(cfg.DATASET.FORMAT_TRAIN_SET):
                imgs_image.append(os.path.join(im_dir, f))
        return imgs_image

    if len(dir) == 3:
        im_dir = str(dir[0])
        scr_dir = str(dir[1])
        edge_dir = str(dir[2])
        imgs_image = []
        imgs_scr = []
        imgs_edge = []
        for f in os.listdir(im_dir):
            if f.endswith(cfg.DATASET.FORMAT_TRAIN_SET):
                imgs_image.append(os.path.join(im_dir, f))
        for f in os.listdir(scr_dir):
            if f.endswith(cfg.DATASET.FORMAT_MASK):
                imgs_scr.append(os.path.join(scr_dir, f))
        for f in os.listdir(edge_dir):
            if f.endswith(cfg.DATASET.FORMAT_MASK):
                imgs_edge.append(os.path.join(edge_dir, f))
        return imgs_image, imgs_scr, imgs_edge


class MyDataSet(Dataset):
    def __init__(self, dataset_dir, aug_is_need=True, p=None):
        im_dir = os.path.join(dataset_dir, "image")
        scr_dir = os.path.join(dataset_dir, "scribble")
        edge_dir = os.path.join(dataset_dir, "edge")

        mean_std = io.loadmat(cfg.DATASET.MEAN_STD)
        mean = mean_std['mean'][0]
        std = mean_std['std'][0]

        self.train_img_size = cfg.TRAIN.TRAIN_IMG_SIZE

        self.fetch3_transform = mytf_fetch3.Compose(
            mytf_fetch3.Normalize(mean=mean, std=std),
            mytf_fetch3.Resize(self.train_img_size, self.train_img_size),
            mytf_fetch3.RandomHorizontalFlip(),
            # tf.RandomCrop(288, 288),
            mytf_fetch3.ToTensor()
        )

        self.im_list, self.scr_list, self.edge_list = _make_dataset(im_dir, scr_dir, edge_dir)

    def __getitem__(self, index):
        # load image
        im_filepath = self.im_list[index]
        scr_filepath = self.scr_list[index]
        edge_filepath = self.edge_list[index]
        x = cv2.imread(im_filepath).astype(np.float32)

        im = cv2.imread(im_filepath).astype(np.float32)[:, :, ::-1]
        scr = cv2.imread(scr_filepath).astype(np.float32)[:, :, ::-1]
        edge = cv2.imread(edge_filepath).astype(np.float32)[:, :, ::-1]

        if im is None or scr is None:
            print(im_filepath)

        im, scr, edge = self.fetch3_transform(im, scr, edge)

        mask = scr.clone()
        gt = mask.clone()

        mask[mask == 2.0] = 255.0
        mask[mask == 1.0] = 255.0
        mask = mask / 255

        gt[gt == 2.0] = 0.
        gt[gt == 1.0] = 255.0
        gt = gt / 255

        edge = edge / 255

        return im, mask, gt, edge

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

        mean_std = io.loadmat(cfg.DATASET.MEAN_STD)
        mean = mean_std['mean'][0]
        std = mean_std['std'][0]

        self.transform = mytf_fetch1.Compose(
            mytf_fetch1.Normalize(mean=mean, std=std),
            mytf_fetch1.Resize(test_size, test_size),
            mytf_fetch1.ToTensor()
        )

        self.im_list = _make_dataset(im_dir)

    def __getitem__(self, index):
        im_path = self.im_list[index]

        image = cv2.imread(im_path).astype(np.float32)[:, :, ::-1]
        H, W, C = image.shape

        image = self.transform(image)

        name = self.im_list[index].split('\\')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        return image, (H, W), name, im_path

    def __len__(self):
        return len(self.im_list)


def load_test_data():
    dataset = MyTestDataSet(dataset_dir=cfg.DATASET.TEST_SET)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)
    logger.warning('Test set: total {} images in dir: {}'.format(dataset.__len__(), cfg.DATASET.TEST_SET))

    return dataloader
