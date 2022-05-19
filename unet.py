import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
from PIL import Image

from resnet import resnet50
from vgg16 import VGG16

import colorsys
import copy
import os

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, inputs1, inputs2):

        outputs = torch.cat([inputs1, self.up(inputs2)],dim=1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, pretrained=False, backbone='vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained, in_channels=in_channels)
            in_filters = [192, 384, 768, 1024]
        elif backbone == 'resnet50':
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone, use vgg, resnet50')
        out_filters = [64, 128, 256, 512]
        # upsampling
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.RelU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.RelU(),
            )
        else:
            self.up_conv = None

        #final conv layer without any concat
        self.last_conv = nn.Conv2d(out_filters[0], num_classes, kernel_size=1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            feat1 = self.vgg.features[:4](inputs)
            feat2 = self.vgg.features[4:9](feat1)
            feat3 = self.vgg.features[9:16](feat2)
            feat4 = self.vgg.features[16:23](feat3)
            feat5 = self.vgg.features[23:-1](feat4)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.last_conv(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == 'vgg':
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == 'resnet50':
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == 'vgg':
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == 'resnet50':
            for param in self.resnet.parameters():
                param.requires_grad = True

    def letterbox_image(image, size):
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image

    #---------------------------#
    #   detect a single image   #
    #---------------------------#
    def detect_image(self, image):
        old_image = copy.deepcopy(image)
        
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]

        #resize by adding grey paddings without distorting the image
        image, nw, nh = self.letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        #add the batch_size dimension to feed into network
        images = [np.array(image)/255]
        #permute so the channel dimension is at the front
        images = np.transpose(images, (0, 3, 1, 2))

        with torch.no_grad():
            images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.sofmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)

            #the next line removes the grey padding added before
            pr = pr[int((self.model_image_size[0]-nh)//2): int((self.model_image_size[0]-nw)//2+nh), int((self.model_image_size[1]-nw)//2): int((self.model_image_size[1]-nw)//2+nw)]

        # assign a color to each class label
        seg_img = np.zeros(np.shape(pr)[0],np.shape(pr)[1], order=3)
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr[:,:] == c) * (self.colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:,:] == c) * (self.colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:,:] == c) * (self.colors[c][2])).astype('uint8')

        image = Image.fromarray(np.uint8(seg_img)).resize((original_w, original_h))
        if self.blend: #blend the segmentation image with the original image
            image = Image.blend(old_img, image, 0.7)

        return image