from config.config import cfg
import os
from core.libs.tools import detach_cudatensor

import scipy.io as io
import torch
from PIL import Image
from torchvision.transforms import transforms as tf
import cv2
import numpy as np


def get_test_im_name(im_dir):
    """
    Get the complete test-image file name
    @param im_dir: complete test-image dir
    @return: complete test-image file name list
    """
    file_name = []
    im_dir = os.path.join(im_dir, 'image')
    for f in os.listdir(im_dir):
        if f.endswith(cfg.DATASET.FORMAT_TEST_SET):
            fname, _ = f.split(cfg.DATASET.FORMAT_TEST_SET)
            file_name.append(fname)
    return file_name


def to_tensor(im):
    mean_std = io.loadmat(cfg.DATASET.MEAN_STD)
    mean = mean_std['mean'][0]
    std = mean_std['std'][0]
    im = tf.Compose([tf.ToTensor(), tf.Normalize(mean=mean, std=std)])(im)
    im = torch.unsqueeze(im, 0).cuda()
    return im


def load_image(filepath, file_name, k):
    fname = os.path.join(filepath, 'image', '{}-{}.tif'.format(file_name, k))
    im = Image.open(fname)
    im = to_tensor(im)
    return im


def save_acc_to_txt(dir, acc_log):
    with open(dir, 'a') as f:
        f.writelines(acc_log)
        f.write('\n')


def load_patch_info(filepath, file_name):
    patch_info = io.loadmat(os.path.join(filepath, 'image', file_name + '_patch_info.mat'))
    if 'centre_win' in patch_info:
        m, n, overlap, rows, cols, patch_size, centre_win = patch_info['m'][0][0], \
                                                            patch_info['n'][0][0], \
                                                            patch_info['overlap'][0][0], \
                                                            patch_info['rows'][0][0], \
                                                            patch_info['cols'][0][0], \
                                                            patch_info['patch_size'][0][0], \
                                                            patch_info['centre_win'][0][0]
        return m, n, overlap, rows, cols, patch_size, centre_win
    else:
        m, n, rows, cols, patch_size = patch_info['m'][0][0], \
                                       patch_info['n'][0][0], \
                                       patch_info['rows'][0][0], \
                                       patch_info['cols'][0][0], \
                                       patch_info['patch_size'][0][0]
        return m, n, rows, cols, patch_size


class SaveInferRes:
    def __init__(self, pred, gt, save_dir, fname=None):
        '''
        Save infer results
        :param pred_p: prediction probability map
        :param pred: predicted binary map
        :param gt: reference label
        :param save_dir: save dir
        :param fname: image name
        '''

        if cfg.MODEL.NUM_CLASSES == 1:
            pred = torch.nn.Sigmoid()(pred).squeeze().cpu().detach().numpy()
            self.pred_p = np.uint8(pred * 255)
            _, pred = cv2.threshold(pred, thresh=0.5, maxval=1, type=cv2.THRESH_BINARY)
            pred = np.uint8(pred * 255)
            self.pred = pred
        else:
            pred = pred.squeeze()  # N*H*W
            pred = torch.nn.Softmax(dim=0)(pred)
            pred_p = pred[1, :, :].cpu().detach().numpy()
            self.pred_p = np.uint8(pred_p * 255)
            pred = torch.argmax(pred, dim=0).cpu().detach().numpy()
            self.pred = np.uint8(pred * 255)

        self.gt = detach_cudatensor(gt)

        self.save_dir = save_dir

        if fname is not None:
            imsrc = os.path.join(cfg.TEST.COMPLETE_TEST_IMAGE_DIR, 'image', fname + cfg.DATASET.FORMAT)
            self.im = cv2.imread(imsrc)
            if self.im is None:
                imsrc = os.path.join(cfg.DATASET.TEST_SET, 'image', fname + cfg.DATASET.FORMAT)
                self.im = cv2.imread(imsrc)
            self.fname = fname

    def _save_pred_p(self):
        out = os.path.join(self.save_dir, self.fname + '.png')
        pred = np.uint8(self.pred_p)
        cv2.imwrite(out, pred)

    def _save_pred_binary(self):
        out = os.path.join(self.save_dir, self.fname + '-binary.tif')
        cv2.imwrite(out, self.pred)

    def __generate_alpha_mask(self, pred, color):
        mask = np.zeros(self.im.shape, dtype=np.uint8)

        r, g, b = mask[:, :, 2], mask[:, :, 1], mask[:, :, 0]
        r[pred == 255] = color[0]
        g[pred == 255] = color[1]
        b[pred == 255] = color[2]
        mask[:, :, 2], mask[:, :, 1], mask[:, :, 0] = r, g, b

        return mask

    def _save_alpha_pred(self):
        # _, pred = cv2.threshold(self.pred, thresh=cfg.SEG_THRESHOLD, maxval=255, type=cv2.THRESH_BINARY)
        pred = np.uint8(self.pred)

        temp = np.zeros(pred.shape).astype('uint8')

        # color = [0, 81, 235]
        color = [0, 255, 255]
        TP = np.zeros(pred.shape).astype('uint8')
        TP[:, :] = np.where((pred[:, :] == 255) & (self.gt[:, :] == 255), 255, 0)
        temp += TP
        TP = self.__generate_alpha_mask(pred=TP, color=color)

        color = [255, 214, 0]
        FP = np.zeros(pred.shape).astype('uint8')
        FP[:, :] = np.where((pred[:, :] == 255) & (self.gt[:, :] == 0), 255, 0)
        temp += FP
        FP = self.__generate_alpha_mask(pred=FP, color=color)

        color = [252, 0, 187]
        FN = np.zeros(pred.shape).astype('uint8')
        FN[:, :] = np.where((pred[:, :] == 0) & (self.gt[:, :] == 255), 255, 0)
        temp += FN
        FN = self.__generate_alpha_mask(pred=FN, color=color)

        mask = FN + TP + FP
        im_rgba = Image.fromarray(self.im).convert('RGBA')
        mask = Image.fromarray(mask).convert('RGBA')

        im_rgba = Image.blend(im_rgba, mask, 0.5)

        im_rgba = np.array(im_rgba)

        self.im[:, :, 0] = np.where(temp == 255, im_rgba[:, :, 0], self.im[:, :, 0])
        self.im[:, :, 1] = np.where(temp == 255, im_rgba[:, :, 1], self.im[:, :, 1])
        self.im[:, :, 2] = np.where(temp == 255, im_rgba[:, :, 2], self.im[:, :, 2])

        out = os.path.join(self.save_dir, self.fname + '-alpha.tif')
        cv2.imwrite(out, self.im)

    def main(self):
        self._save_pred_p()
        # self._save_alpha_pred()


class Resize(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, image):
        if not isinstance(image, torch.Tensor):
            raise TypeError('Input image should be torch.Tensor, get {}'.format(type(image)))

        _, _, rows, cols = image.size()

        new_rows, new_cols = int(self.scale * rows), int(self.scale * cols)

        new_image = torch.nn.functional.interpolate(image, size=(new_rows, new_cols), mode='bilinear',
                                                    align_corners=True)
        return new_image


class Flip(object):
    def __call__(self, image):
        if not isinstance(image, torch.Tensor):
            raise TypeError('Input image should be torch.Tensor, get {}'.format(type(image)))

        new_image = torch.flip(image, [3])

        return new_image
