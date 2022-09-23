# -*- coding: utf-8 -*-

import os

# Config and Network model
from config.config import cfg
from model.networks.SSBANet import SSBANet_Vgg16 as net

# Trainer and Inference
from core.train.trainer_SSBANet import Trainer
from core.inference.infer import Inference

model = net().cuda()


def train():
    Trainer(model=model).run()


def infer():
    Inference(well_trained_model=model).run()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if cfg.IS_TRAIN:
        train()
    else:
        infer()
