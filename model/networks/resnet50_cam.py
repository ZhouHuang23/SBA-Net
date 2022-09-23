import torch.nn as nn
import torch.nn.functional as F
from core.libs import torchutils
from model.backbone import ResNet50
import torch

class CAM(nn.Module):

    def __init__(self):
        super().__init__()
        self.k = 1e-3
        self.resnet50 = ResNet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, 17, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)

        # generate attention map
        kernel = 1
        attention = F.avg_pool2d(x, kernel, stride=1, padding=kernel // 2, count_include_pad=False)
        attention = (attention * self.k)
        attention = torch.softmax(attention.view(attention.shape[0], attention.shape[1], -1), dim=-1).view(
            attention.shape)
        attention = attention.detach() * (attention.shape[-2] * attention.shape[-1])
        x = x * attention

        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, 17)

        return x

    def infer(self, x):
        """ in inferring stage, the attention module is removed.
        parameter x: [2, C, W, H], two tensors are original image and flipped image.
        """
        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)

        x = F.conv2d(x, self.classifier.weight)
        x = nn.functional.relu(x)
        x = x[0] + x[1].flip(-1)

        return x.detach()

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return list(self.backbone.parameters()), list(self.newly_added.parameters())


