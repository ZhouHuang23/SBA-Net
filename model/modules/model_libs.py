# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from core.libs import set_logger
from config.config import cfg

logger = set_logger()


def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class ModuleHelper:
    @staticmethod
    def BNReLU(num_features, inplace=True):
        return nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=inplace)
        )

    @staticmethod
    def BatchNorm2d(num_features):
        return nn.BatchNorm2d(num_features)

    @staticmethod
    def Conv3x3_BNReLU(in_channels, out_channels, stride=1, dilation=1, groups=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                      groups=groups, bias=False),
            ModuleHelper.BNReLU(out_channels)
        )

    @staticmethod
    def Conv1x1_BNReLU(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            ModuleHelper.BNReLU(out_channels)
        )

    @staticmethod
    def Conv1x1(in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)


class Conv3x3(nn.Module):
    def __init__(self, in_chs, out_chs, dilation=1, dropout=None):
        super(Conv3x3, self).__init__()

        if dropout is None:
            self.conv = ModuleHelper.Conv3x3_BNReLU(in_channels=in_chs, out_channels=out_chs, dilation=dilation)
        else:
            self.conv = nn.Sequential(
                ModuleHelper.Conv3x3_BNReLU(in_channels=in_chs, out_channels=out_chs, dilation=dilation),
                nn.Dropout(dropout)
            )

        initialize_weights(self.conv)

    def forward(self, x):
        return self.conv(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=False),
            ModuleHelper.BNReLU(out_planes)
        )

    def forward(self, x):
        return self.conv_bn_relu(x)


class Conv1x1(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(Conv1x1, self).__init__()

        self.conv = ModuleHelper.Conv1x1_BNReLU(in_chs, out_chs)

        initialize_weights(self.conv)

    def forward(self, x):
        return self.conv(x)


# depthwise separable convolution
class DepSepConvolutions(nn.Module):
    def __init__(self, in_chs, out_chs, dilation=1):
        super(DepSepConvolutions, self).__init__()

        # depth wise
        self.DW = ModuleHelper.Conv3x3_BNReLU(in_channels=in_chs, out_channels=in_chs, dilation=dilation, groups=in_chs)
        # point wise
        self.PW = ModuleHelper.Conv1x1_BNReLU(in_channels=in_chs, out_channels=out_chs)

        initialize_weights(self.DW, self.PW)

    def forward(self, x):
        y = self.DW(x)
        y = self.PW(y)

        return y


class DecoderBlock(nn.Module):
    def __init__(self, in_chs, out_chs, dropout=0.1):
        super(DecoderBlock, self).__init__()

        self.doubel_conv = nn.Sequential(
            Conv3x3(in_chs=in_chs, out_chs=out_chs, dropout=dropout),
            Conv3x3(in_chs=out_chs, out_chs=out_chs, dropout=dropout)
        )

        initialize_weights(self.doubel_conv)

    def forward(self, x):
        out = self.doubel_conv(x)
        return out


class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(Classifier, self).__init__()

        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        initialize_weights(self.conv)

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class DoubleAttentionModule(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(DoubleAttentionModule, self).__init__()

        self.db = DecoderBlock(in_chs, out_chs)
        self.ca = ChannelAttention(out_chs)
        self.sa = SpatialAttention()

    def forward(self, x):
        out_db = self.db(x)
        out_ca = self.ca(out_db) * out_db
        out_sa = self.sa(out_db) * out_db

        return out_ca + out_sa


class ResBlock(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1):
        super(ResBlock, self).__init__()
        ## conv branch
        self.left = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chs)
        )
        ## shortcut branch
        self.short_cut = nn.Sequential()
        if stride != 1 or in_chs != out_chs:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chs))

    ### get the residual
    def forward(self, x):
        return F.relu(self.left(x) + self.short_cut(x))


class BoundaryEnhancementModule(nn.Module):
    def __init__(self, in_chs=3, out_chs=128):
        super(BoundaryEnhancementModule, self).__init__()
        self.horizontal_conv = nn.Sequential(
            nn.Conv2d(in_chs, 128, (1, 7)),
            ModuleHelper.BNReLU(128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            Conv1x1(128, 1)
        )  # bs,1,352,346
        self.conv1x1_h = Conv1x1(2, 8)

        self.vertical_conv = nn.Sequential(
            nn.Conv2d(in_chs, 128, (7, 1)),
            ModuleHelper.BNReLU(128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            Conv1x1(128, 1)
        )
        self.conv1x1_v = Conv1x1(2, 8)
        self.conv_out = Conv1x1(16, out_chs)

    def forward(self, x):
        bs, chl, w, h = x.size()[0], x.size()[1], x.size()[2], x.size()[3]
        x_h = self.horizontal_conv(x)
        x_h = Upsample(x_h, (w, h))
        x_v = self.vertical_conv(x)
        x_v = Upsample(x_v, (w, h))
        x_arr = x.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny = np.zeros((bs, 1, w, h))
        for i in range(bs):
            canny[i] = cv2.Canny(x_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()

        h_canny = torch.cat((x_h, canny), dim=1)
        v_canny = torch.cat((x_v, canny), dim=1)
        h_v_canny = torch.cat((self.conv1x1_h(h_canny), self.conv1x1_v(v_canny)), dim=1)
        h_v_canny_out = self.conv_out(h_v_canny)

        return h_v_canny_out

        # position attention


class PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def Upsample(x, size):
    return nn.functional.interpolate(x, size, mode='bilinear', align_corners=True)


class GateUnit(nn.Module):
    def __init__(self, in_chs):
        super(GateUnit, self).__init__()

        self.conv = nn.Conv2d(in_chs, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        initialize_weights(self.conv)

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)

        return y


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // 16, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        initialize_weights(self.fc1, self.fc2)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out) * x
        return out


class Aux_Module(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(Aux_Module, self).__init__()

        self.aux = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        initialize_weights(self.aux)

    def forward(self, x):
        res = self.aux(x)
        return res


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=16,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Edge_Module(nn.Module):

    def __init__(self, in_fea=[64, 256, 512], mid_fea=32):
        super(Edge_Module, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_fea[0], mid_fea, 1)
        self.conv4 = nn.Conv2d(in_fea[1], mid_fea, 1)
        self.conv5 = nn.Conv2d(in_fea[2], mid_fea, 1)
        self.conv5_2 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_4 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_5 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)

        self.classifer = nn.Conv2d(mid_fea * 3, 1, kernel_size=3, padding=1)
        self.rcab = RCAB(mid_fea * 3)

    def forward(self, x0, x1, x2):
        _, _, h, w = x0.size()  # 352*352
        edge2_fea = self.relu(self.conv2(x0))
        edge2 = self.relu(self.conv5_2(edge2_fea))
        edge4_fea = self.relu(self.conv4(x1))
        edge4 = self.relu(self.conv5_4(edge4_fea))
        edge5_fea = self.relu(self.conv5(x2))
        edge5 = self.relu(self.conv5_5(edge5_fea))

        edge4 = F.interpolate(edge4, size=(h, w), mode='bilinear', align_corners=True)  # 352*352
        edge5 = F.interpolate(edge5, size=(h, w), mode='bilinear', align_corners=True)  # 352*352

        edge = torch.cat([edge2, edge4, edge5], dim=1)  # 352*352*96
        edge = self.rcab(edge)
        edge = self.classifer(edge)  # 16*1*352*352
        return edge


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, x, s_edge):  # x: bs,192,22,22 fused_semantic_boundary_s:bs, 1, 352, 352
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear', align_corners=True)
        out = img_features

        edge_features = F.interpolate(s_edge, x_size[2:],
                                      mode='bilinear', align_corners=True)
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class RFB_modified(nn.Module):
    def __init__(self, in_chl, out_chl):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            Conv1x1(in_chl, out_chl)
        )

        self.branch1 = nn.Sequential(
            Conv1x1(in_chl, out_chl),
            BasicConv2d(out_chl, out_chl, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_chl, out_chl, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_chl, out_chl, 3, padding=3, dilation=3)
        )

        self.branch2 = nn.Sequential(
            Conv1x1(in_chl, out_chl),
            BasicConv2d(out_chl, out_chl, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_chl, out_chl, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_chl, out_chl, 3, padding=5, dilation=5)
        )

        self.branch3 = nn.Sequential(
            Conv1x1(in_chl, out_chl),
            BasicConv2d(out_chl, out_chl, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_chl, out_chl, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_chl, out_chl, 3, padding=7, dilation=7)
        )

        self.conv_cat = Conv3x3(4 * out_chl, out_chl)
        self.conv_res = Conv1x1(in_chl, out_chl)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = self.relu(self.conv_cat(torch.cat((x0, x1, x2, x3), 1)) + self.conv_res(x))
        return x


class Dense_Aggregation_input_5(nn.Module):
    def __init__(self, chl):
        super(Dense_Aggregation_input_5, self).__init__()

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_0 = Conv3x3(chl, chl)
        self.conv2_0 = Conv3x3(chl, chl)
        self.conv3_0 = Conv3x3(chl, chl)
        self.conv4_0 = Conv3x3(chl, chl)
        self.conv5_0 = Conv3x3(chl, chl)

        self.conv1_1 = Conv3x3(chl, chl)
        self.conv2_1 = Conv3x3(chl, chl)
        self.conv3_1 = Conv3x3(chl, chl)
        self.conv4_1 = Conv3x3(chl * 2, chl * 2)

        self.conv1_2 = Conv3x3(chl, chl)
        self.conv2_2 = Conv3x3(chl, chl)
        self.conv3_2 = Conv3x3(chl * 3, chl * 3)

        self.conv1_3 = Conv3x3(chl, chl)
        self.conv2_3 = Conv3x3(chl * 4, chl * 4)

        self.conv_cat_1_0 = Conv3x3(chl * 2, chl * 2)
        self.conv_cat_2_1 = Conv3x3(chl * 3, chl * 3)
        self.conv_cat_3_2 = Conv3x3(chl * 4, chl * 4)
        self.conv_cat_4_3 = Conv3x3(chl * 5, chl * 5)

        self.conv_d1 = Conv3x3(chl * 5, chl * 5)
        self.conv_d2 = nn.Conv2d(chl * 5, 1, 1)

    def forward(self, x0, x1, x2, x3, x4):  # x0:bs,32,22,22 x1:bs,32,44,44 x2:bs,32,88,88 x3:bs,32,196,196
        x0_1 = x0  # bs,32,22,22
        x1_1 = self.conv1_0(self.up2(x0)) * x1  # bs,32,44,44
        x2_1 = self.conv2_0(self.up2(self.up2(x0))) * self.conv1_1(self.up2(x1)) * x2  # bs,32,88,88
        x3_1 = self.conv3_0(self.up2(self.up2(self.up2(x0)))) * self.conv2_1(self.up2(self.up2(x1))) * self.conv1_2(
            self.up2(x2)) * x3  # bs,32,196,196
        x4_1 = self.conv4_0(self.up2(self.up2(self.up2(self.up2(x0))))) * self.conv3_1(
            self.up2(self.up2(self.up2(x1)))) * self.conv2_2(self.up2(self.up2(x2))) * self.conv1_3(
            self.up2(x3)) * x4  # bs,32,352,352

        x1_2 = self.conv_cat_1_0(torch.cat((x1_1, self.conv5_0(self.up2(x0_1))), 1))  # bs,64,44,44
        x2_3 = self.conv_cat_2_1(torch.cat((x2_1, self.conv4_1(self.up2(x1_2))), 1))  # bs,96,88,88
        x3_4 = self.conv_cat_3_2(torch.cat((x3_1, self.conv3_2(self.up2(x2_3))), 1))  # bs,128,196,196
        x4_5 = self.conv_cat_4_3(torch.cat((x4_1, self.conv2_3(self.up2(x3_4))), 1))  # bs,160,352,352

        x = self.conv_d1(x4_5)
        x = self.conv_d2(x)

        return x


class Dense_Aggregation_input_3(nn.Module):
    def __init__(self, chl):
        super(Dense_Aggregation_input_3, self).__init__()
        self.relu = nn.ReLU(True)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4_1 = Conv3x3(chl, chl)
        self.conv4_2 = Conv3x3(chl, chl)
        self.conv4_3 = Conv3x3(chl, chl)

        self.conv3_1 = Conv3x3(chl, chl)
        self.conv3_2 = Conv3x3(chl*2, chl*2)

        self.conv_cat_4_3 = Conv3x3(chl * 2, chl * 2)
        self.conv_cat_3_2 = Conv3x3(chl * 3, chl * 3)

        self.conv_out = Conv3x3(chl * 3, chl * 3)

    def forward(self, x2, x3, x4):
        x4_1 = x4
        x4_2 = self.conv4_1(self.up2(x4)) * x3
        x4_3 = self.conv4_2(self.up2(self.up2(x4))) * self.conv3_1(self.up2(x3)) * x2

        x43_cat = self.conv_cat_4_3(torch.cat((x4_2, self.conv4_3(self.up2(x4_1))), 1))
        x32_cat = self.conv_cat_3_2(torch.cat((x4_3, self.conv3_2(self.up2(x43_cat))), 1))

        out = self.conv_out(x32_cat)

        return out


if __name__ == '__main__':
    ras = BoundaryEnhancementModule(3, 8).cuda()
    input_tensor = torch.randn(2, 3, 352, 352).cuda()

    out = ras(input_tensor)
    x = out
    # from torchstat import stat
    #
    # model = DoubleAttentionModule(64, 128)
    # stat(model, (3, 352, 352))
