#https://github.com/ShawnXYang/Face-Robustness-Benchmark

import os
import sys
# sys.path.append('networks/Face_Robustness_Benchmark_Pytorch')

import torch
from torch import nn
from torch.nn import functional as F
from networks.Face_Robustness_Benchmark_Pytorch.networks.ResNet import ResNet_50
import cv2

class Resnet50(nn.Module):

    def __init__(self, device='cuda', input_shape=(112, 112), ckpt='ckpts/Face_Robustness_Benchmark_Pytorch/Backbone_ResNet_50_Epoch_36_Batch_204696_Time_2019-04-14-14-44_checkpoint.pth'):
        super(Resnet50, self).__init__()
        self.model = ResNet_50(input_shape)
        self.model.feature = True
        self.model.load_state_dict(torch.load(ckpt))
        self.model.eval().to(device)
        self.input_shape = input_shape
        self.name = 'ResNet50'

    def preprocess(self, x):
        # for an rgb [0, 255], any shape, (N, C, H, W) images, how to preprocess to feed into model
        assert x.size(1) == 3 or x.size(1) == 1
        if x.shape[2:] != self.input_shape:
            x = F.interpolate(x, self.input_shape, mode='bilinear', align_corners=True)
        x = (x - 127.5) / 128
        return x

    def forward(self, x):
        # must be normalized
        out = self.model(self.preprocess(x))
        return out / (torch.sqrt(torch.sum(out ** 2, dim=1, keepdims=True)) + 1e-5)



if __name__ == "__main__":
    model = Resnet50()
    img = '/home/gaoxianfeng/data/face_evasion_pytorch/data/ant_aligned/source/1.JPG'
    img = cv2.imread(img)[:, :, ::-1].copy().transpose([2, 0, 1])
    img1 = torch.cuda.FloatTensor(img).unsqueeze(0)
    # print(img.shape)

    img = '/home/gaoxianfeng/data/face_evasion_pytorch/data/ant_aligned/source/2.JPG'
    img = cv2.imread(img)[:, :, ::-1].copy().transpose([2, 0, 1])
    img2 = torch.cuda.FloatTensor(img).unsqueeze(0)

    embedding = model(torch.cat([img1, img2], 0))

    print(torch.sum(embedding[0] * embedding[1]))
