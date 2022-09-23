import pandas as pd
import torch

def print_train_hyper_params(cfg, logger):
    train_info = {'Model': cfg.MODEL.NAME,
                  'Optimizer': cfg.TRAIN.OPTIMIZER,
                  'Lr_scheduler': cfg.TRAIN.LR_SCHEDULE + ' (Power={})'.format(cfg.TRAIN.LR_POWER),
                  'Device': cfg.TRAIN.DEVICE,
                  'Dataset': cfg.DATASET.NAME,
                  }

    info = pd.DataFrame(data=train_info, index=[0]).T

    logger.warning(info)
    logger.warning(
        'Training hyper-params: lr={}, weight_decay={}, epochs={}, batch_size={}'.format(cfg.TRAIN.BASE_LR,
                                                                                         cfg.TRAIN.WEIGHT_DECAY,
                                                                                         cfg.TRAIN.EPOCHS,
                                                                                         cfg.TRAIN.BATCH_SIZE
                                                                                         ))
    logger.warning('Training hyper-params: aug_strategy={}, aug_p={}'.format(cfg.TRAIN.AUG_STRATEGY, cfg.TRAIN.AUG_P))


def parser_optimizer(cfg, model):
    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(lr=cfg.TRAIN.BASE_LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                     params=model.parameters())
        return optimizer
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(weight_decay=cfg.TRAIN.WEIGHT_DECAY, params=model.parameters(),
                                    lr=cfg.TRAIN.BASE_LR, momentum=0.9)
        return optimizer
    else:
        assert "Optimizer is not supported"


def parser_lr_schedule(cfg, optimizer, dt_size=None):
    if cfg.TRAIN.LR_SCHEDULE == 'exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=cfg.TRAIN.LR_POWER)
        return lr_scheduler
    elif cfg.TRAIN.LR_SCHEDULE == 'poly':
        if dt_size is None:
            assert "data size is not given."
        max_iterations = cfg.TRAIN.EPOCHS * dt_size
        lr_scheduler = torch.optim.lr_scheduler.PolyLR(optimizer, max_iterations, power=0.9, last_epoch=-1)
        return lr_scheduler


def epoch_visualize(curr_epoch, writer, train_score, val_score):
    # train
    # writer.add_scalar('Train/Loss', train_score[0], curr_epoch)
    # # writer.add_scalar('Train/Precision', train_score[1], curr_epoch)
    # # writer.add_scalar('Train/Recall', train_score[2], curr_epoch)
    # writer.add_scalar('Train/F1-Score', train_score[3], curr_epoch)
    # writer.add_scalar('Train/Accuracy', train_score[4], curr_epoch)
    # writer.add_scalar('Train/Iou', train_score[5], curr_epoch)
    # writer.add_scalar('Train/lr_rate', train_score[6], curr_epoch)
    #
    # # validate
    # writer.add_scalar('Validate/Loss', val_score[0], curr_epoch)
    # # writer.add_scalar('Validate/Precision', val_score[1], curr_epoch)
    # # writer.add_scalar('Validate/Recall', val_score[2], curr_epoch)
    # writer.add_scalar('Validate/F1-Score', val_score[3], curr_epoch)
    # writer.add_scalar('Validate/Accuracy', val_score[4], curr_epoch)
    # writer.add_scalar('Validate/Iou', val_score[5], curr_epoch)

    writer.add_scalars('Train/Loss', {'Train': train_score[0], 'Validate': val_score[0]}, curr_epoch)
    writer.add_scalars('Train/Precision', {'Train': train_score[1], 'Validate': val_score[1]}, curr_epoch)
    writer.add_scalars('Train/Accuracy', {'Train': train_score[2], 'Validate': val_score[2]}, curr_epoch)
    writer.add_scalars('Train/F1-Score', {'Train': train_score[3], 'Validate': val_score[3]}, curr_epoch)
    writer.add_scalars('Train/Accuracy', {'Train': train_score[4], 'Validate': val_score[4]}, curr_epoch)
    writer.add_scalars('Train/IoU', {'Train': train_score[5], 'Validate': val_score[5]}, curr_epoch)
    writer.add_scalar('Train/lr_rate', train_score[-2], curr_epoch)
    writer.add_scalars('Train/Elapsed Time', {'Train': train_score[-1], 'Validate': val_score[-1]}, curr_epoch)


def train_epoch_visualize(curr_epoch, writer, train_score):
    # write training log
    writer.add_scalar('Train/Loss', train_score[0], curr_epoch)
    writer.add_scalar('Train/lr_rate', train_score[-2], curr_epoch)
