# -*- coding: utf-8 -*-
import os
import shutil

import torch
import xlrd
import xlwt
from PIL import Image
from matplotlib import pyplot as plt

from config.config import cfg


def mk_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mk_dirs_r(file_path):
    if os.path.exists(file_path):
        shutil.rmtree(file_path, ignore_errors=True)
    os.makedirs(file_path)


def rm_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def detach_cudatensor(pred):
    """将GPU Tensor 转为 CPU Tensor"""
    if isinstance(pred, torch.cuda.FloatTensor) or isinstance(pred, torch.FloatTensor) or isinstance(pred, torch.Tensor):
        if pred is not None:
            pred = torch.squeeze(pred)  # 去除为1的维度
            pred = pred.cpu().detach().numpy()
        else:
            assert "Prediction is None"

        pred = pred.astype('float32')

    return pred


class Visualization:
    def __init__(self, src1, src2):
        """
        plot loss, acc, iou...etc
        :param src1: saved train score path
        :param src2: saved validate score path
        """
        workbook1 = xlrd.open_workbook(src1)
        workbook2 = xlrd.open_workbook(src2)
        self.sheet1 = workbook1.sheet_by_index(0)
        self.sheet2 = workbook2.sheet_by_index(0)
        self.font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}

    def _plot(self, k, ylabel, title, legend=["train", "validate"]):
        y1, y2 = np.zeros(self.sheet1.nrows), np.zeros(self.sheet1.nrows)
        x = range(self.sheet1.nrows)
        plt.figure()
        for i in x:
            y1[i] = self.sheet1.cell_value(i, k)
            y2[i] = self.sheet2.cell_value(i, k)

        plt.plot(x, y1, 'r')
        plt.plot(x, y2, 'b')
        plt.legend(legend)
        plt.xlabel("epoch", self.font)
        plt.ylabel(ylabel, self.font)
        plt.title(title, self.font)
        plt.show()

    def plot_loss(self):
        k = 0
        self._plot(k, ylabel='Loss', title='Loss')

    def plot_ber(self):
        k = 3
        self._plot(k, ylabel='BER', title='BER')

    def plot_f1_score(self):
        k = 6
        self._plot(k, ylabel='F1-Score', title='F1-Score')

    def plot_acc(self):
        k = 7
        self._plot(k, ylabel='Overall Accuracy', title='Overall Accuracy')

    def plot_iou(self):
        k = 8
        self._plot(k, ylabel='IOU', title='IOU')

    def plot_te(self):
        k = 11
        self._plot(k, ylabel='Total error', title='Total error')


def epoch_visualize(epoch, writer, train_score, val_score):
    epoch -= 1
    # train
    writer.add_scalar('Train/Loss', train_score[0], epoch + 1)
    writer.add_scalar('Train/Precision', train_score[1], epoch + 1)
    writer.add_scalar('Train/Recall', train_score[2], epoch + 1)
    writer.add_scalar('Train/F1-Score', train_score[3], epoch + 1)
    writer.add_scalar('Train/Accuracy', train_score[4], epoch + 1)
    writer.add_scalar('Train/Iou', train_score[5], epoch + 1)
    writer.add_scalar('Train/lr_rate', train_score[6], epoch + 1)

    # validate
    writer.add_scalar('Validate/Loss', val_score[0], epoch + 1)
    writer.add_scalar('Validate/Precision', val_score[1], epoch + 1)
    writer.add_scalar('Validate/Recall', val_score[2], epoch + 1)
    writer.add_scalar('Validate/F1-Score', val_score[3], epoch + 1)
    writer.add_scalar('Validate/Accuracy', val_score[4], epoch + 1)
    writer.add_scalar('Validate/Iou', val_score[5], epoch + 1)


def iter_train_visualize(writer, train_score, itr):
    writer.add_scalar('Train/Loss', train_score[0], itr + 1)
    writer.add_scalar('Train/Precision', train_score[1], itr + 1)
    writer.add_scalar('Train/Recall', train_score[2], itr + 1)
    writer.add_scalar('Train/F1-Score', train_score[3], itr + 1)
    writer.add_scalar('Train/Accuracy', train_score[4], itr + 1)
    writer.add_scalar('Train/Iou', train_score[5], itr + 1)


def iter_val_visualize(writer, val_score, itr):
    writer.add_scalar('Val/Loss', val_score[0], itr + 1)
    writer.add_scalar('Val/Precision', val_score[1], itr + 1)
    writer.add_scalar('Val/Recall', val_score[2], itr + 1)
    writer.add_scalar('Val/F1-Score', val_score[3], itr + 1)
    writer.add_scalar('Val/Accuracy', val_score[4], itr + 1)
    writer.add_scalar('Val/Iou', val_score[5], itr + 1)


def init_train_xls(worksheet):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = 'Times New Roman'
    font.bold = True
    style.font = font
    worksheet.write(0, 0, 'Loss', style)
    worksheet.write(0, 1, 'Precision', style)
    worksheet.write(0, 2, 'Recall', style)
    worksheet.write(0, 3, 'F1-score', style)
    worksheet.write(0, 4, 'OA', style)
    worksheet.write(0, 5, 'IoU', style)
    worksheet.write(0, 6, 'lr_rate', style)
    return worksheet


def init_val_xls(worksheet):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = 'Times New Roman'
    font.bold = True
    style.font = font
    worksheet.write(0, 0, 'Loss', style)
    worksheet.write(0, 1, 'Precision', style)
    worksheet.write(0, 2, 'Recall', style)
    worksheet.write(0, 3, 'F1-score', style)
    worksheet.write(0, 4, 'OA', style)
    worksheet.write(0, 5, 'IoU', style)

    return worksheet


def init_test_xls(worksheet):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = 'Times New Roman'
    font.bold = True
    style.font = font
    worksheet.write(0, 0, 'Image', style)
    worksheet.write(0, 1, 'Precision', style)
    worksheet.write(0, 2, 'Recall', style)
    worksheet.write(0, 3, 'F1-score', style)
    worksheet.write(0, 4, 'OA', style)
    worksheet.write(0, 5, 'IoU', style)

    return worksheet



