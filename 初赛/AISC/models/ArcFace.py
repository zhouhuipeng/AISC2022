# https://github.com/ronghuaiyang/arcface-pytorch.git
# TODO: make it work

import os
import sys
# sys.path.append('./')

import numpy as np
import torch
import torchvision
import cv2

from networks.arcface_pytorch.models import resnet_face18
from torch import nn
from torch.nn import functional as F

class Gray(object):
    # convert an RGB image into a gray image
    def __init__(self, device='cuda'):
        channel_weights = [0.299, 0.587, 0.114]
        channel_weights = torch.FloatTensor(channel_weights).to(device)
        self.channel_weights = channel_weights.view(1, -1, 1, 1)

    def __call__(self, x):
        # TODO: make efficient
        return torch.sum(self.channel_weights * x, dim=1, keepdims=True)

class ArcFace(nn.Module):
    
    def __init__(self, device='cuda', input_shape=(128, 128), ckpt='ckpts/ArcFace_pytorch/resnet18_110.pth'):
        super(ArcFace, self).__init__()
        self.model = resnet_face18(False)
        self.model.load_state_dict(torch.load(ckpt))
        self.model.eval().to(device)
        self.input_shape = input_shape
        self.converter = Gray(device)
        self.name = 'ArcFace'

    def preprocess(self, x):
        # for an rgb [0, 255], any shape, (N, C, H, W) images, how to preprocess to feed into model 
        assert x.size(1) == 3 or x.size(1) == 1
        if x.size(1) == 3:
            # to gray
            x = self.converter(x)
        
        if x.shape[2:] != self.input_shape:
            x = F.interpolate(x, self.input_shape, mode='bilinear', align_corners=True)
        
        x = (x - 127.5) / 128
        return x

    def forward(self, x):
        # must be normalized 
        out = self.model(self.preprocess(x))
        return out / torch.sqrt((torch.sum(out ** 2, dim=1, keepdims=True) + 1e-5))

if __name__ == "__main__":
    
    model = ArcFace()
    img = '/home/gaoxianfeng/data/face_evasion_pytorch/data/ant_aligned/source/1.JPG'
    img = cv2.imread(img)[:, :, ::-1].copy().transpose([2, 0, 1])
    img1 = torch.cuda.FloatTensor(img).unsqueeze(0)
    # print(img.shape)

    img = '/home/gaoxianfeng/data/face_evasion_pytorch/data/ant_aligned/source/2.JPG'
    img = cv2.imread(img)[:, :, ::-1].copy().transpose([2, 0, 1])
    img2 = torch.cuda.FloatTensor(img).unsqueeze(0)

    embedding = model(torch.cat([img1, img2], 0))

    print(torch.sum(embedding[0] * embedding[1]))
