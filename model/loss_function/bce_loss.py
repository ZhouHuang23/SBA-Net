# -*- coding: utf-8 -*-
import torch
from torch.nn import modules as nn


class SAWBCELoss(nn.Module):
    def __init__(self):
        super(SAWBCELoss, self).__init__()

    def forward(self, pred, gt):
        eposion = 1e-10
        count_pos = torch.sum(gt) * 1.0 + eposion
        count_neg = torch.sum(1.0 - gt) * 1.0
        beta = count_neg / count_pos
        # beta_back = count_pos / (count_pos + count_neg)
        loss = nn.BCEWithLogitsLoss(pos_weight=beta)(pred, gt)

        return loss


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, pred, gt):
        loss = nn.BCEWithLogitsLoss()(pred, gt)
        return loss


class OhemBCELoss(nn.Module):
    def __init__(self, thresh=0.7):
        super(OhemBCELoss, self).__init__()

        self.thresh = float(thresh)

        self.criterion = nn.BCELoss()

    def forward(self, pred, target):
        N, C, H, W = target.size()

        # normalize
        pred = nn.Sigmoid()(pred)

        pred_copy = pred
        ones = torch.ones_like(pred)

        pred = torch.where(pred > self.thresh, ones, pred_copy)

        loss = self.criterion(pred, target)
