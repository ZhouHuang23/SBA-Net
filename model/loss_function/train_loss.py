import torch.nn.modules as nn
import torch
from config.config import cfg
import torch.nn.functional as F


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()

    @staticmethod
    def get_weights_bce(target):
        eposion = 1e-10
        count_pos = torch.sum(target) * 1.0 + eposion
        count_neg = torch.sum(1.0 - target) * 1.0
        beta = count_neg / count_pos
        return beta

    @staticmethod
    def get_weights_ce(target):
        count_pos = torch.sum(target)
        B, H, W = target.size()
        beta = float(count_pos) / float((H * W * B))
        return beta

    def forward(self, segbody_pred, segbody_target, loss_weight=None):
        # p2: aspp, p1: decoder, p0:fusion pred

        if cfg.MODEL.NUM_CLASSES == 1:
            beta = self.get_weights_bce(segbody_target)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=beta)
        else:
            beta = self.get_weights_ce(segbody_target)
            weight = torch.FloatTensor([1.0 - beta, beta]).cuda()
            loss_fn = nn.CrossEntropyLoss(weight=weight)

        loss = []
        for pred in segbody_pred:
            loss.append(loss_fn(pred, segbody_target))

        if len(segbody_pred) == 1:
            final_loss = loss[0]
            return final_loss
        else:
            final_loss = 0.0
            if loss_weight is None:
                raise TypeError('loss_weight get None')
            for i in range(len(segbody_pred)):
                final_loss = final_loss + loss[i] * loss_weight[i]
            return final_loss


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()


class Hybrid_E_Loss(nn.Module):
    def __init__(self):
        super(Hybrid_E_Loss, self).__init__()

    def forward(self, pred, mask):
        """
        Introduction:
            Hybrid Eloss
        Paramaters:
            :param pred:
            :param mask:
            :return:
        Usage:
            loss = hybrid_e_loss(prediction_map, gt_mask)
            loss.backward()
        """
        # adaptive weighting masks
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

        # weighted binary cross entropy loss function
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = ((weit * wbce).sum(dim=(2, 3)) + 1e-8) / (weit.sum(dim=(2, 3)) + 1e-8)

        # weighted e loss function
        pred = torch.sigmoid(pred)
        mpred = pred.mean(dim=(2, 3)).view(pred.shape[0], pred.shape[1], 1, 1).repeat(1, 1, pred.shape[2],
                                                                                      pred.shape[3])
        phiFM = pred - mpred

        mmask = mask.mean(dim=(2, 3)).view(mask.shape[0], mask.shape[1], 1, 1).repeat(1, 1, mask.shape[2],
                                                                                      mask.shape[3])
        phiGT = mask - mmask

        EFM = (2.0 * phiFM * phiGT + 1e-8) / (phiFM * phiFM + phiGT * phiGT + 1e-8)
        QFM = (1 + EFM) * (1 + EFM) / 4.0
        eloss = 1.0 - QFM.mean(dim=(2, 3))

        # weighted iou loss function
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1.0 - (inter + 1 + 1e-8) / (union - inter + 1 + 1e-8)

        return (wbce + eloss + wiou).mean()


class LocalSaliencyCoherence(torch.nn.Module):
    """
    This loss function based on the following paper.
    Please consider using the following bibtex for citation:
    @article{obukhov2019gated,
        author={Anton Obukhov and Stamatios Georgoulis and Dengxin Dai and Luc {Van Gool}},
        title={Gated {CRF} Loss for Weakly Supervised Semantic Image Segmentation},
        journal={CoRR},
        volume={abs/1906.04651},
        year={2019},
        url={http://arxiv.org/abs/1906.04651},
    }
    """

    def forward(
            self, y_hat_softmax, kernels_desc, kernels_radius, sample, height_input, width_input,
            mask_src=None, mask_dst=None, compatibility=None, custom_modality_downsamplers=None, out_kernels_vis=False
    ):
        """
        Performs the forward pass of the loss.
        :param y_hat_softmax: A tensor of predicted per-pixel class probabilities of size NxCxHxW
        :param kernels_desc: A list of dictionaries, each describing one Gaussian kernel composition from modalities.
            The final kernel is a weighted sum of individual kernels. Following example is a composition of
            RGBXY and XY kernels:
            kernels_desc: [{
                'weight': 0.9,          # Weight of RGBXY kernel
                'xy': 6,                # Sigma for XY
                'rgb': 0.1,             # Sigma for RGB
            },{
                'weight': 0.1,          # Weight of XY kernel
                'xy': 6,                # Sigma for XY
            }]
        :param kernels_radius: Defines size of bounding box region around each pixel in which the kernel is constructed.
        :param sample: A dictionary with modalities (except 'xy') used in kernels_desc parameter. Each of the provided
            modalities is allowed to be larger than the shape of y_hat_softmax, in such case downsampling will be
            invoked. Default downsampling method is area resize; this can be overriden by setting.
            custom_modality_downsamplers parameter.
        :param width_input, height_input: Dimensions of the full scale resolution of modalities
        :param mask_src: (optional) Source mask.
        :param mask_dst: (optional) Destination mask.
        :param compatibility: (optional) Classes compatibility matrix, defaults to Potts model.
        :param custom_modality_downsamplers: A dictionary of modality downsampling functions.
        :param out_kernels_vis: Whether to return a tensor with kernels visualized with some step.
        :return: Loss function value.
        """
        assert y_hat_softmax.dim() == 4, 'Prediction must be a NCHW batch'
        N, C, height_pred, width_pred = y_hat_softmax.shape

        device = y_hat_softmax.device

        assert width_input % width_pred == 0 and height_input % height_pred == 0 and \
               width_input * height_pred == height_input * width_pred, \
            f'[{width_input}x{height_input}] !~= [{width_pred}x{height_pred}]'

        kernels = self._create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers
        )

        y_hat_unfolded = self._unfold(y_hat_softmax, kernels_radius)
        y_hat_unfolded = torch.abs(
            y_hat_unfolded[:, :, kernels_radius, kernels_radius, :, :].view(N, C, 1, 1, height_pred,
                                                                            width_pred) - y_hat_unfolded)

        loss = torch.mean(
            (kernels * y_hat_unfolded).view(N, C, (kernels_radius * 2 + 1) ** 2, height_pred, width_pred).sum(dim=2,
                                                                                                              keepdim=True))

        out = {
            'loss': loss.mean(),
        }

        if out_kernels_vis:
            out['kernels_vis'] = self._visualize_kernels(
                kernels, kernels_radius, height_input, width_input, height_pred, width_pred
            )

        return out

    @staticmethod
    def _downsample(img, modality, height_dst, width_dst, custom_modality_downsamplers):
        if custom_modality_downsamplers is not None and modality in custom_modality_downsamplers:
            f_down = custom_modality_downsamplers[modality]
        else:
            f_down = F.adaptive_avg_pool2d
        return f_down(img, (height_dst, width_dst))

    @staticmethod
    def _create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers
    ):
        kernels = None
        for i, desc in enumerate(kernels_desc):
            weight = desc['weight']
            features = []
            for modality, sigma in desc.items():
                if modality == 'weight':
                    continue
                if modality == 'xy':
                    feature = LocalSaliencyCoherence._get_mesh(N, height_pred, width_pred, device)
                else:
                    assert modality in sample, \
                        f'Modality {modality} is listed in {i}-th kernel descriptor, but not present in the sample'
                    feature = sample[modality]
                    # feature = LocalSaliencyCoherence._downsample(
                    #     feature, modality, height_pred, width_pred, custom_modality_downsamplers
                    # )
                feature /= sigma
                features.append(feature)
            features = torch.cat(features, dim=1)
            kernel = weight * LocalSaliencyCoherence._create_kernels_from_features(features, kernels_radius)
            kernels = kernel if kernels is None else kernel + kernels
        return kernels

    @staticmethod
    def _create_kernels_from_features(features, radius):
        assert features.dim() == 4, 'Features must be a NCHW batch'
        N, C, H, W = features.shape
        kernels = LocalSaliencyCoherence._unfold(features, radius)
        kernels = kernels - kernels[:, :, radius, radius, :, :].view(N, C, 1, 1, H, W)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()
        # kernels[:, :, radius, radius, :, :] = 0
        return kernels

    @staticmethod
    def _get_mesh(N, H, W, device):
        return torch.cat((
            torch.arange(0, W, 1, dtype=torch.float32, device=device).view(1, 1, 1, W).repeat(N, 1, H, 1),
            torch.arange(0, H, 1, dtype=torch.float32, device=device).view(1, 1, H, 1).repeat(N, 1, 1, W)
        ), 1)

    @staticmethod
    def _unfold(img, radius):
        assert img.dim() == 4, 'Unfolding requires NCHW batch'
        N, C, H, W = img.shape
        diameter = 2 * radius + 1
        return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)

    @staticmethod
    def _visualize_kernels(kernels, radius, height_input, width_input, height_pred, width_pred):
        diameter = 2 * radius + 1
        vis = kernels[:, :, :, :, radius::diameter, radius::diameter]
        vis_nh, vis_nw = vis.shape[-2:]
        vis = vis.permute(0, 1, 4, 2, 5, 3).contiguous().view(kernels.shape[0], 1, diameter * vis_nh, diameter * vis_nw)
        if vis.shape[2] > height_pred:
            vis = vis[:, :, :height_pred, :]
        if vis.shape[3] > width_pred:
            vis = vis[:, :, :, :width_pred]
        if vis.shape[2:] != (height_pred, width_pred):
            vis = F.pad(vis, [0, width_pred - vis.shape[3], 0, height_pred - vis.shape[2]])
        vis = F.interpolate(vis, (height_input, width_input), mode='nearest')
        return vis


def laplacian_edge(img):
    laplacian_filter = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filter = torch.reshape(laplacian_filter, [1, 1, 3, 3])
    filter = filter.cuda()
    lap_edge = F.conv2d(img, filter, stride=1, padding=1)
    return lap_edge


def gradient_x(img):
    sobel = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter = torch.reshape(sobel, [1, 1, 3, 3])
    filter = filter.cuda()
    gx = F.conv2d(img, filter, stride=1, padding=1)
    return gx


def gradient_y(img):
    sobel = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filter = torch.reshape(sobel, [1, 1, 3, 3])
    filter = filter.cuda()
    gy = F.conv2d(img, filter, stride=1, padding=1)
    return gy


def charbonnier_penalty(s):
    cp_s = torch.pow(torch.pow(s, 2) + 0.001 ** 2, 0.5)
    return cp_s


def get_saliency_smoothness(pred, gt, size_average=True):
    alpha = 10
    s1 = 10
    s2 = 1
    ## first oder derivative: sobel
    sal_x = torch.abs(gradient_x(pred))
    sal_y = torch.abs(gradient_y(pred))
    gt_x = gradient_x(gt)
    gt_y = gradient_y(gt)
    w_x = torch.exp(torch.abs(gt_x) * (-alpha))
    w_y = torch.exp(torch.abs(gt_y) * (-alpha))
    cps_x = charbonnier_penalty(sal_x * w_x)
    cps_y = charbonnier_penalty(sal_y * w_y)
    cps_xy = cps_x + cps_y

    ## second order derivative: laplacian
    lap_sal = torch.abs(laplacian_edge(pred))
    lap_gt = torch.abs(laplacian_edge(gt))
    weight_lap = torch.exp(lap_gt * (-alpha))
    weighted_lap = charbonnier_penalty(lap_sal * weight_lap)

    smooth_loss = s1 * torch.mean(cps_xy) + s2 * torch.mean(weighted_lap)

    return smooth_loss


class smoothness_loss(torch.nn.Module):
    def __init__(self, size_average=True):
        super(smoothness_loss, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return get_saliency_smoothness(pred, target, self.size_average)
