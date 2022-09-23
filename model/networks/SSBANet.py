import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.VGG import Backbone_VGG16_in3

from model.modules.model_libs import Edge_Module, _AtrousSpatialPyramidPoolingModule, RCAB, Conv3x3, Classifier, \
    DoubleAttentionModule, BoundaryEnhancementModule, RFB_modified, Dense_Aggregation_input_3


class SSBANet_Vgg16(nn.Module):
    def __init__(self, channel=32):
        super(SSBANet_Vgg16, self).__init__()
        (
            self.encoder1,
            self.encoder2,
            self.encoder4,
            self.encoder8,
            self.encoder16,
        ) = Backbone_VGG16_in3()

        self.relu = nn.ReLU(inplace=True)

        self.dense_agg_3 = Dense_Aggregation_input_3(channel * 2)

        self.edge_lager = Edge_Module()
        self.aspp = _AtrousSpatialPyramidPoolingModule(192, 64, output_stride=16)
        self.rcab_feat = RCAB(channel * 6)
        self.sal_conv = nn.Conv2d(1, channel, kernel_size=3, padding=1, bias=False)
        self.edge_conv = nn.Conv2d(1, channel, kernel_size=3, padding=1, bias=False)
        self.rcab_sal_edge = RCAB(channel * 2)

        self.aspp_conv = nn.Conv2d(384, 64, kernel_size=1, bias=False)

        # A1
        self.agg1_conv = Conv3x3(192, 64)
        self.rcab_agg1 = RCAB(64)

        # A2
        self.agg2_conv = Conv3x3(192, 64)
        self.rcab_agg2 = RCAB(64)

        # A3
        self.agg3_conv = Conv3x3(192, 64)
        self.rcab_agg3 = RCAB(64)

        # A4
        self.agg4_conv = Conv3x3(192, 64)
        self.rcab_agg4 = RCAB(64)

        self.final_sal_seg = nn.Sequential(
            Conv3x3(channel * 2, channel),
            Conv3x3(channel, channel),
            Classifier(channel, 1)
        )
        self.fuse_semantic_boundary = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.fused_edge_sal = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False)

        self.rfb0 = RFB_modified(64, 64)
        self.rfb1 = RFB_modified(128, 64)
        self.rfb2 = RFB_modified(256, 64)
        self.rfb3 = RFB_modified(512, 64)
        self.rfb4 = RFB_modified(512, 64)

        self.DA0 = DoubleAttentionModule(64, channel)
        self.DA1 = DoubleAttentionModule(64, channel)

        self.BEM = BoundaryEnhancementModule(in_chs=3, out_chs=channel)

        self.fc_boundary_out = nn.Conv2d(channel * 3, 1, 1, bias=True)

    def forward(self, x):
        x_size = x.size()

        x0 = self.encoder1(x)
        x1 = self.encoder2(x0)
        x2 = self.encoder4(x1)
        x3 = self.encoder8(x2)
        x4 = self.encoder16(x3)

        x0_rbf = self.rfb0(x0)
        x1_rbf = self.rfb1(x1)
        x2_rbf = self.rfb2(x2)
        x3_rbf = self.rfb3(x3)
        x4_rbf = self.rfb4(x4)

        # =====boundary detection branch =====
        da0 = self.DA0(x0_rbf)
        da1 = self.DA1(x1_rbf)
        da0_ = F.interpolate(da0, x_size[2:], mode='bilinear', align_corners=True)
        da1_ = F.interpolate(da1, x_size[2:], mode='bilinear', align_corners=True)
        bem_out = self.BEM(x)
        boundary_out = self.fc_boundary_out(torch.cat((da0_, da1_, bem_out), dim=1))
        boundary_out_s = torch.sigmoid(boundary_out)

        da3 = self.dense_agg_3(x2_rbf, x3_rbf, x4_rbf)
        aspp_out = self.aspp(da3, boundary_out_s)
        aspp_out = self.aspp_conv(aspp_out)

        # A1: x4_rbf x3_rbf
        x4_rbf_up = F.interpolate(x4_rbf, x3_rbf.size()[2:], mode='bilinear', align_corners=True)
        aspp_out_up = F.interpolate(aspp_out, x3_rbf.size()[2:], mode='bilinear', align_corners=True)
        agg1 = self.agg1_conv(torch.cat((x4_rbf_up, x3_rbf, aspp_out_up), 1))  # bs, 64, 44, 44
        agg1 = self.rcab_agg1(agg1)

        # A2: agg1 x2_rbf
        agg1_up = F.interpolate(agg1, x2_rbf.size()[2:], mode='bilinear', align_corners=True)
        aspp_out_up = F.interpolate(aspp_out, x2_rbf.size()[2:], mode='bilinear', align_corners=True)
        agg2 = self.agg2_conv(torch.cat((agg1_up, x2_rbf, aspp_out_up), 1))  # bs, 64, 44, 44
        agg2 = self.rcab_agg2(agg2)

        # A3: agg2 x1_rbf
        agg2_up = F.interpolate(agg2, x1_rbf.size()[2:], mode='bilinear', align_corners=True)
        aspp_out_up = F.interpolate(aspp_out, x1_rbf.size()[2:], mode='bilinear', align_corners=True)
        agg3 = self.agg3_conv(torch.cat((agg2_up, x1_rbf, aspp_out_up), 1))  # bs, 64, 44, 44
        agg3 = self.rcab_agg3(agg3)

        # A4: agg3 x0_rbf
        agg3_up = F.interpolate(agg3, x0_rbf.size()[2:], mode='bilinear', align_corners=True)
        aspp_out_up = F.interpolate(aspp_out, x0_rbf.size()[2:], mode='bilinear', align_corners=True)
        agg3 = self.agg4_conv(torch.cat((agg3_up, x0_rbf, aspp_out_up), 1))  # bs, 64, 44, 44
        agg4 = self.rcab_agg4(agg3)

        sal = self.final_sal_seg(agg4)

        return boundary_out, sal


if __name__ == '__main__':
    ras = SSBANet_Vgg16().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()
    input_tensor2 = torch.randn(1, 3, 176, 176).cuda()

    out = ras(input_tensor2)
