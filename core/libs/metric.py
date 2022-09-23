import torch
import numpy as np
from core.libs.tools import detach_cudatensor
import cv2
from config.config import cfg


def save_acc_score(epoch, score, worksheet, workbook, fname):
    for i in range(len(score)):
        worksheet.write(epoch + 1, i, score[i])
    workbook.save(fname)


# Cal Acc
class Accuracy:
    def __init__(self):
        self.loss = []
        self.Pre = []
        self.Rec = []
        self.F1_score = []
        self.Acc = []
        self.Iou = []

    def _cal_acc(self, pred, mask):
        """
        cal. acc. subFun.
        """
        if cfg.MODEL.NUM_CLASSES == 1:
            pred = torch.nn.Sigmoid()(pred)
            pred = detach_cudatensor(pred)
            mask = detach_cudatensor(mask)
            _, pred = cv2.threshold(pred, thresh=0.5, maxval=1, type=cv2.THRESH_BINARY)
        else:
            pred = torch.nn.Softmax(dim=0)(pred)
            pred = torch.argmax(pred, dim=0)
            pred = detach_cudatensor(pred).astype(np.uint8)
            mask = detach_cudatensor(mask)

        if np.max(mask.flatten()) == 255:
            mask = np.uint8(mask / 255)

        N = np.sum(mask.flatten())
        # no buildings in the reference maps
        if N == 0:
            TP = np.sum(np.logical_and(np.equal(mask, 0), np.equal(pred, 0)))
            FP = np.sum(np.logical_and(np.equal(mask, 1), np.equal(pred, 0)))
            TN = np.sum(np.logical_and(np.equal(mask, 1), np.equal(pred, 1)))
            FN = np.sum(np.logical_and(np.equal(mask, 0), np.equal(pred, 1)))
        else:
            TP = np.sum(np.logical_and(np.equal(mask, 1), np.equal(pred, 1)))
            FP = np.sum(np.logical_and(np.equal(mask, 0), np.equal(pred, 1)))
            TN = np.sum(np.logical_and(np.equal(mask, 0), np.equal(pred, 0)))
            FN = np.sum(np.logical_and(np.equal(mask, 1), np.equal(pred, 0)))

        eps = 1e-10
        Precision = TP / (TP + FP + eps)
        Recall = TP / (TP + FN + eps)
        F1_score = 2 * (Precision * Recall) / (Precision + Recall + eps)
        Acc = (TP + TN) / (TP + FP + TN + FN + eps)
        Iou = TP / (TP + FP + FN + eps)

        self.Pre.append(Precision)
        self.Rec.append(Recall)
        self.F1_score.append(F1_score)
        self.Acc.append(Acc)
        self.Iou.append(Iou)

        return Precision, Recall, F1_score, Acc, Iou

    def cal_mini_batch_acc(self, pred, target, loss):

        batch_size = pred.size()[0]
        pre1, rc1, f11, acc1, iou1 = [], [], [], [], []
        for i in range(batch_size):
            x = pred[i, :, :, :]
            y = target[i, :, :]

            Precision, Recall, F1_score, Acc, Iou = self._cal_acc(pred=x, mask=y)

            pre1.append(Precision)
            rc1.append(Recall)
            f11.append(F1_score)
            acc1.append(Acc)
            iou1.append(Iou)

        pre = np.nanmean(np.array(pre1))
        rc = np.nanmean(np.array(rc1))
        f1 = np.nanmean(np.array(f11))
        acc = np.nanmean(np.array(acc1))
        iou = np.nanmean(np.array(iou1))

        self.loss.append(loss)  # append minibatch loss

        logs = "Loss={:.4f}, Pre={:.4f}, Rec={:.4f}, F1_score={:.4f}, Acc={:.4f}, Iou={:.4f}".format(loss, pre, rc,
                                                                                                     f1, acc, iou)
        return logs

    def cal_test_batch_acc(self, pred, target):
        """Compute accuracy for every test image"""
        pred = torch.squeeze(pred)
        pre, rc, f1, acc, iou = self._cal_acc(pred=pred, mask=target)
        log = "Pre={:.4f}, Rec={:.4f}, F1_score={:.4f}, Acc={:.4f}, Iou={:.4f}".format(pre, rc, f1, acc, iou)
        return log, pre, rc, f1, acc, iou

    def cal_test_acc(self):
        pre = np.nanmean(np.array(self.Pre))
        rc = np.nanmean(np.array(self.Rec))
        f1 = np.nanmean(np.array(self.F1_score))
        acc = np.nanmean(np.array(self.Acc))
        iou = np.nanmean(np.array(self.Iou))

        log = "Pre={:.4f}, Rec={:.4f}, F1_score={:.4f}, Acc={:.4f}, Iou={:.4f}".format(pre, rc, f1, acc, iou)

        return log

    def cal_train_epoch_acc(self):
        loss = np.nanmean(np.array(self.loss))
        pre = np.nanmean(np.array(self.Pre))
        rc = np.nanmean(np.array(self.Rec))
        f1 = np.nanmean(np.array(self.F1_score))
        acc = np.nanmean(np.array(self.Acc))
        iou = np.nanmean(np.array(self.Iou))

        epoch_log = "Loss={:.4f}, Pre={:.4f}, Rec={:.4f}, F1_score={:.4f}, Acc={:.4f}, Iou={:.4f}".format(loss, pre, rc,
                                                                                                          f1, acc, iou)

        return [loss, pre, rc, f1, acc, iou], epoch_log

    def cal_val_epoch_acc(self):
        loss = np.nanmean(np.array(self.loss))
        pre = np.nanmean(np.array(self.Pre))
        rc = np.nanmean(np.array(self.Rec))
        f1 = np.nanmean(np.array(self.F1_score))
        acc = np.nanmean(np.array(self.Acc))
        iou = np.nanmean(np.array(self.Iou))

        epoch_log = "Loss={:.4f}, Pre={:.4f}, Rec={:.4f}, F1_score={:.4f}, Acc={:.4f}, Iou={:.4f}".format(loss, pre, rc,
                                                                                                          f1, acc, iou)
        return [loss, pre, rc, f1, acc, iou], epoch_log