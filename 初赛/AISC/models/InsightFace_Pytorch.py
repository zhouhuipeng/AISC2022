# https://github.com/TreB1eN/InsightFace_Pytorch


import os
import sys
# sys.path.append('networks/InsightFace_Pytorch')

import numpy as np
import torch
import torchvision
import cv2

from networks.InsightFace_Pytorch.model import Backbone,  MobileFaceNet, l2_norm
from torch import nn
from torch.nn import functional as F


class MobileFacenet(nn.Module):
    
    def __init__(self, device='cuda', input_shape=(112,112), ckpt='ckpts/InsightFace_Pytorch/model_mobilefacenet.pth'):
        super(MobileFacenet, self).__init__()
        self.model = MobileFaceNet(512)
        self.model.load_state_dict(torch.load(ckpt))
        self.model.eval().to(device)
        self.input_shape = input_shape
        self.name = 'InsightFace_Pytorch_mobilefacenet'

    def preprocess(self, x):
        # for an rgb [0, 255], any shape, (N, C, H, W) images, how to preprocess to feed into model 
        assert x.size(1) == 3 or x.size(1) == 1
        if x.shape[2:] != self.input_shape:
            x = F.interpolate(x, self.input_shape, mode='bilinear', align_corners=True)
        x = (x - 127.5) / 128
        return x

    def forward(self, x):
        # must be normalized 
        return self.model(self.preprocess(x))





if __name__ == "__main__":
    
    model = IR_SE50()
    img = 'data/1.JPG'
    img = cv2.imread(img)[:, :, ::-1].copy().transpose([2, 0, 1])
    img1 = torch.cuda.FloatTensor(img).unsqueeze(0)
    # print(img.shape)

    img = 'data/2.JPG'
    img = cv2.imread(img)[:, :, ::-1].copy().transpose([2, 0, 1])
    img2 = torch.cuda.FloatTensor(img).unsqueeze(0)

    embedding = model(torch.cat([img1, img2], 0))

    print(torch.sum(embedding[0] * embedding[1]))

    
    

    