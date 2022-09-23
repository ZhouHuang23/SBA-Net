import torch
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model.networks.resnet50_cam import CAM

# import voc12.dataloader
from core.libs import pyutils, torchutils
from core.data.data_loader_fetch2 import load_train_CAM_data


def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss')

    model.eval()

    with torch.no_grad():
        for (images, _, _) in data_loader:
            img = images
            bs, chs, H, W = img.size()

            # label = pack['label'].cuda(non_blocking=True)
            label = torch.ones(bs, 1).cuda(non_blocking=True)
            x = model(img)
            loss1 = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss': loss1.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss')))

    return


def run(args):
    model = CAM()
    '''
    original resize_long is set as '(320, 640)', parts of image would be cropped if 'resize_long' is bigger than 
    'crop_size'. To prevent that, 'resize_long' is set as '(320, 512)'.
    '''
    # train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_aug_list, voc12_root=args.voc12_root,
    #                                                             resize_long=(320, 512), hor_flip=True,
    #                                                             crop_size=512, crop_method="random")
    # train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
    #                                shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    train_data_loader, dt_size = load_train_CAM_data()
    max_step = dt_size * args.cam_num_epoches

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep + 1, args.cam_num_epoches))

        for step, (name, img, label) in enumerate(train_data_loader):

            img = img.cuda().float()
            label = label.cuda(non_blocking=True)

            x = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            avg_meter.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        else:
            # validate(model, train_data_loader)
            timer.reset_stage()

    save_path = r'Path\outputs\weights\sess' + r'\res50_cam_attention.pth'
    torch.save(model.module.state_dict(), save_path)
    torch.cuda.empty_cache()
