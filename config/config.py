# -*- coding: utf-8 -*-

import torch
import os
import yaml

dataset_name = 'S-EOR'
model_name = 'SSBANet'

this_dir = os.path.dirname(__file__)

# load yaml file configs
yaml_file_path = os.path.join(this_dir, dataset_name + '.yaml')
if not os.path.exists(yaml_file_path):
    raise FileNotFoundError('Corresponding dataset configuration file ({}.yaml) is not found in config dir ({}).'
                            .format(dataset_name, yaml_file_path))
else:
    yaml_file = open(yaml_file_path, 'r')
    data = yaml.load(yaml_file, Loader=yaml.FullLoader)


# Cuda and cudnn
class CUDNN:
    SEED = 10
    BENCHMARK = True
    DETERMINISTIC = False
    ENABLED = True


# Common params for model
class MODEL:
    NAME = model_name
    BACKBONE = data['MODEL']['BACKBONE']
    PRETRAINED = True
    BACKBONE_OS = data['MODEL']['BACKBONE_OS']
    NUM_CLASSES = data['MODEL']['NUM_CLASSES']


# Loss related params
class LOSS:
    USE_OHEM = False
    if USE_OHEM:
        OHEM_THRES = 0.9
        OHEM_KEEP = 100000
    CLASS_BALANCE_WEIGHT = None


# Dataset related params
class DATASET:
    NAME = dataset_name
    ROOT = r'Path\Weakly-SOD'
    TRAIN_SET = os.path.join(ROOT, NAME, 'train')
    MEAN_STD = os.path.join(ROOT, NAME, NAME + "-mean-std.mat")
    VAL_SET = os.path.join(ROOT, NAME, 'test')  # No Val set
    TEST_SET = os.path.join(ROOT, NAME, 'test')
    FORMAT_TRAIN_SET = '.jpg'
    FORMAT_TEST_SET = '.jpg'
    FORMAT_MASK = '.png'


# Dataloader related params
class DATALOADER:
    NUM_WORKERS = 0
    PIP_MEMORY = True
    AUG_P = data['DATALOADER']['AUG_P']  # augmentation probability for each batch


# training params set
class TRAIN:
    DEVICE = torch.cuda.get_device_name(0)

    # train datasize
    TRAIN_IMG_SIZE = 352

    # optimizer
    BASE_LR = float(data['TRAIN']['BASE_LR'])
    OPTIMIZER = data['TRAIN']['OPTIMIZER']
    WEIGHT_DECAY = float(data['TRAIN']['WEIGHT_DECAY'])

    # learning rate adjust strategy
    LR_SCHEDULE = data['TRAIN']['LR_SCHEDULE']
    LR_POWER = float(data['TRAIN']['LR_POWER'])

    EPOCHS = data['TRAIN']['EPOCHS']
    BATCH_SIZE = data['TRAIN']['BATCH_SIZE']

    DEFAULT_AUG_P = data['TRAIN']['DEFAULT_AUG_P']
    AUG_STRATEGY = data['TRAIN']['AUG_STRATEGY']
    AUG_P = data['TRAIN']['AUG_P']


# Checkpoint related params
class CKPT:
    NUM = data['CKPT']['NUM']
    SAVE_DIR = os.path.join('outputs', 'weights', '{}_{}'.format(MODEL.NAME, DATASET.NAME))
    SELECTED_INFER_CKPT = '20-ckpt.pth'


class LOG:
    SAVE_DIR = os.path.join('outputs', 'run_logs', '{}_{}'.format(MODEL.NAME, DATASET.NAME))


# Infer ralated params
class TEST:
    COMPLETE_TEST_IMAGE_DIR = os.path.join(DATASET.ROOT, DATASET.NAME, 'test', 'image')
    SAVE_DIR1 = os.path.join('test_results', '{}_{}_res/'.format(MODEL.NAME, DATASET.NAME))
    SAVE_DIR2 = os.path.join('test_results', '{}_{}_dp_res/'.format(MODEL.NAME, DATASET.NAME))

    FLIP_TEST = data['TEST']['FLIP_TEST']
    MULTI_SCALE_TEST = data['TEST']['MULTI_SCALE_TEST']
    if MULTI_SCALE_TEST:
        SCALE_LIST = data['TEST']['SCALE_TEST']


class cfg:
    CUDNN = CUDNN
    MODEL = MODEL
    LOSS = LOSS
    DATALOADER = DATALOADER
    DATASET = DATASET
    TRAIN = TRAIN
    CKPT = CKPT
    LOG = LOG
    TEST = TEST

    IS_TRAIN = False
