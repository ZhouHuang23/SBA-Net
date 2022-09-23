from config.config import cfg
from core.train.train_libs import print_train_hyper_params, parser_optimizer, parser_lr_schedule, train_epoch_visualize
from core.data.data_loader_fetch4 import load_train_data
from core.libs import Accuracy, mk_dirs_r, set_logger
from model.loss_function.train_loss import LocalSaliencyCoherence
from model.loss_function.train_loss import smoothness_loss
import torch

torch.manual_seed(cfg.CUDNN.SEED)
torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

import os
import time
import torch.nn.functional as F

logger = set_logger()
scaler = GradScaler()


class Trainer:
    def __init__(self, model):
        print_train_hyper_params(cfg, logger)

        self.model = model.cuda()
        self.optimizer = parser_optimizer(cfg, self.model)

        self.loss_fn = torch.nn.BCELoss().cuda()

        # Ref “2021 Structure-Consistent Weakly Supervised Salient Object Detection with Local Saliency Coherence”
        self.loss_lsc = LocalSaliencyCoherence().cuda()
        self.loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
        self.loss_lsc_radius = 5

        self.smooth_loss = smoothness_loss(size_average=True)
        self.train_loader, self.train_dt_size = load_train_data()

        self.writer = SummaryWriter(cfg.LOG.SAVE_DIR)

        # training initialization
        mk_dirs_r(cfg.LOG.SAVE_DIR)
        mk_dirs_r(cfg.CKPT.SAVE_DIR)

        self.current_epoch = 0
        self.lr_scheduler = parser_lr_schedule(cfg, self.optimizer)

    def __train(self):
        """
            employ the mixed precision strategy to perform train procedure for one epoch
            return: the epoch's training score
        """
        acc = Accuracy()

        self.model.train()

        for itr, (images, masks, gts, edges, grays) in enumerate(self.train_loader):

            images, masks, gts, edges, grays = images.cuda().float(), masks.cuda(), gts.cuda(), edges.cuda(
                non_blocking=True).squeeze(), grays.cuda()

            # forward
            self.optimizer.zero_grad()

            img_size = images.size(2) * images.size(3) * images.size(0)
            ratio = img_size / torch.sum(masks)  # ratio约15.6

            boundary_map, sal_map = self.model(images)

            #  Boundary loss
            boundary_map = boundary_map.squeeze().clamp(min=1e-4, max=1 - 1e-4)
            mask_bg_ground = (edges == 0).type(torch.float32)
            mask_fg_ground = (edges == 50).type(torch.float32)
            mask_boundary = ((edges == 100) | (edges == 150)).type(torch.float32)

            loss_bg_ground = (-torch.log(1 - boundary_map) * mask_bg_ground).sum() / (mask_bg_ground.sum() + 1)
            loss_fg_ground = (-torch.log(1 - boundary_map) * mask_fg_ground).sum() / (mask_fg_ground.sum() + 1)
            loss_ground = (loss_fg_ground + loss_bg_ground) / 2
            loss_boundary = (-torch.log(boundary_map) * torch.pow(boundary_map.detach(), 0.5) * mask_boundary).sum() / (
                    mask_boundary.sum() + 1)

            edge_loss = (loss_ground + loss_boundary)

            # Scribble supervision
            sal_ref_prob = torch.sigmoid(sal_map)
            sal_ref_prob_ = sal_ref_prob * masks

            smoothLoss_sal_ref = 0.3 * self.smooth_loss(torch.sigmoid(sal_map), grays)
            loss_ref = ratio * self.loss_fn(sal_ref_prob_, gts * masks) + smoothLoss_sal_ref

            # LSC loss
            image_s = F.interpolate(images, scale_factor=0.25, mode='bilinear', align_corners=True)
            sample = {'rgb': image_s}

            sal_ref_prob_s = F.interpolate(sal_ref_prob, scale_factor=0.25, mode='bilinear', align_corners=True)
            loss_ref_lsc = \
                self.loss_lsc(sal_ref_prob_s, self.loss_lsc_kernels_desc_defaults, self.loss_lsc_radius, sample,
                              image_s.shape[2],
                              image_s.shape[3])['loss']

            loss = edge_loss + loss_ref + loss_ref_lsc

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            if cfg.TRAIN.LR_SCHEDULE.upper() == 'poly'.upper():
                self.lr_scheduler.step()

            batch_log = acc.cal_mini_batch_acc(pred=sal_ref_prob, target=gts, loss=loss.item())
            batch_log = "Training epochs:{}/{}, steps:{}/{}, {}".format(self.current_epoch, cfg.TRAIN.EPOCHS, itr + 1,
                                                                        self.train_dt_size, batch_log)
            logger.debug(batch_log)

        return acc.cal_train_epoch_acc()

    def run(self):
        tic = time.time()

        logger.info('Start training...')

        for epoch in range(cfg.TRAIN.EPOCHS):
            self.current_epoch += 1

            "training one epoch"
            # ------------------------------------------------------------------------------------------
            start_time = time.time()
            train_score, train_logs = self.__train()
            end_time = time.time()

            # generate logs
            train_score.append(self.optimizer.state_dict()['param_groups'][0]['lr'])
            epoch_train_time = (end_time - start_time) / 60
            train_score.append(epoch_train_time)
            train_logs = "Training epochs:{}/{}, {}, elapsed time:{:.2f} min".format(self.current_epoch,
                                                                                     cfg.TRAIN.EPOCHS,
                                                                                     train_logs,
                                                                                     epoch_train_time)
            """display and save training and validation logs"""
            logger.debug('--' * 60)
            logger.warning('Epoch training logs:')
            logger.debug(train_logs)
            logger.debug('--' * 60)

            train_epoch_visualize(curr_epoch=self.current_epoch, writer=self.writer, train_score=train_score)

            if self.current_epoch >= cfg.CKPT.NUM:
                # if self.current_epoch % 10 == 0:
                weight_path = os.path.join(cfg.CKPT.SAVE_DIR, '%d-ckpt.pth' % self.current_epoch)
                all_states = {"net": self.model.state_dict(), cfg.TRAIN.OPTIMIZER: self.optimizer.state_dict(),
                              "epoch": epoch}
                torch.save(obj=all_states, f=weight_path)

                # adjust learning rate after each epoch complete train-eval procedure
                if cfg.TRAIN.LR_SCHEDULE.upper() != 'poly'.upper():
                    self.lr_scheduler.step()

        toc = time.time()
        logger.error('<<<<<<<<<<<<<< End Training >>>>>>>>>>>>>>')
        logger.info('Total elapsed time is {:.2f} hours.'.format((toc - tic) / 3600))
