import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import GaussianBlur

class ResNetFeatureExtractor(torch.nn.Module):
    """多尺度ResNet特征提取器"""
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.layer0 = torch.nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # 1/4尺度
        self.layer2 = resnet.layer2   # 1/8尺度
        self.layer3 = resnet.layer3   # 1/16尺度
        self.layer4 = resnet.layer4   # 1/32尺度

    def forward(self, x):
        x = self.layer0(x)
        f1 = self.layer1(x)   # [B,256,H/4,W/4]
        f2 = self.layer2(f1)   # [B,512,H/8,W/8]
        f3 = self.layer3(f2)   # [B,1024,H/16,W/16]
        f4 = self.layer4(f3)   # [B,2048,H/32,W/32]
        return f4

class blur_loss(nn.Module):
    def __init__(self):
        super(blur_loss, self).__init__()
        self.cnn_feat = ResNetFeatureExtractor()
        self.blur = GaussianBlur(kernel_size=25, sigma=(1.9, 1.9))

    def forward(self, I_om, I_gen):
        return F.mse_loss(self.cnn_feat(I_om), self.cnn_feat(self.blur(I_gen)))