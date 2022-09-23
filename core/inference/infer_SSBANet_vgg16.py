"""
This script is used to do the general prediction without padding.
"""

from config.config import cfg
from core.libs import set_logger, mk_dirs_r, Accuracy, save_acc_score, init_test_xls
from core.data.data_loader_fetch4 import load_test_data
import os
import torch
import numpy as np
import xlwt
import cv2
import torch.nn.functional as F

logger = set_logger()
test_workbook = xlwt.Workbook()
test_worksheet = init_test_xls(test_workbook.add_sheet('test accuracy', cell_overwrite_ok=False))
acc_score = os.path.join(cfg.TEST.SAVE_DIR1, '{}_{}_infer.xls'.format(cfg.MODEL.NAME, cfg.DATASET.NAME))
acc_score_txt = os.path.join(cfg.TEST.SAVE_DIR1, '{}_{}_infer.txt'.format(cfg.MODEL.NAME, cfg.DATASET.NAME))


def infer_TestNet(model):
    logger.info('Start inference....')

    mk_dirs_r(cfg.TEST.SAVE_DIR1)
    test_loader = load_test_data()

    def visualize_edge(pred, image_name):
        for kk in range(pred.shape[0]):
            pred_edge_kk = pred[kk, :, :, :]
            pred_edge_kk = pred_edge_kk[0].detach().cpu().numpy()
            pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
            pred_edge_kk *= 255.0
            pred_edge_kk = pred_edge_kk.astype(np.uint8)
            out_name = cfg.TEST.SAVE_DIR1 + image_name[0]
            cv2.imwrite(out_name, pred_edge_kk)

    with torch.no_grad():
        for image, (H, W), name, im_path in test_loader:
            image = image.cuda().float()

            boundary_map, sal_map = model(image)

            # Save saliency map
            res = sal_map
            res = F.upsample(res, size=(H, W), mode='bilinear', align_corners=False)
            res = (torch.sigmoid(res[0, 0])).cpu().numpy()
            # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res *= 255.0
            out_name = cfg.TEST.SAVE_DIR1 + name[0]
            cv2.imwrite(out_name, np.uint8(res))

            # Save boundary map
            # boundary_map = boundary_map.squeeze().clamp(min=1e-4, max=1 - 1e-4)
            # boundary_map=boundary_map.unsqueeze(0)
            # visualize_edge(torch.sigmoid(boundary_map.unsqueeze(1)),name)
