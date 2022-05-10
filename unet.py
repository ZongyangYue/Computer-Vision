import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
    
    def forward(self, inputs1, inputs2):

        outputs = torch.cat([inputs1, self.up(inputs2)],dim=1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, pretrained=False):
        super(Unet, self).__init__()
        self.vgg = VGG16(pretrained=pretrained, in_channels=in_channels)