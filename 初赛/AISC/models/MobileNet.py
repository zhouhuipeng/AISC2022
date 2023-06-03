#https://github.com/ShawnXYang/Face-Robustness-Benchmark

import os
import sys
# sys.path.append('networks/Face_Robustness_Benchmark_Pytorch')

import torch
from torch import nn
from torch.nn import functional as F
from networks.Face_Robustness_Benchmark_Pytorch.networks.Mobilenet import MobileNet
import cv2

class MobileNetV2(nn.Module):

    def __init__(self, device='cuda', input_shape=(112, 112), ckpt='ckpts/Face_Robustness_Benchmark_Pytorch/Backbone_Mobilenet_Epoch_125_Batch_710750_Time_2019-04-14-18-15_checkpoint.pth'):
        super(MobileNetV2, self).__init__()
        self.model = MobileNet(2)
        self.model.feature = True
        self.model.load_state_dict(torch.load(ckpt))
        self.model.eval().to(device)
        self.input_shape = input_shape
        self.name = 'MobileNet'

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
        return out / torch.sqrt((torch.sum(out ** 2, dim=1, keepdims=True) + 1e-5))


if __name__ == "__main__":
    model = MobileNetV2()
    img = '/home/gaoxianfeng/data/face_evasion_pytorch/data/ant_aligned/source/1.JPG'
    img = cv2.imread(img)[:, :, ::-1].copy().transpose([2, 0, 1])
    img1 = torch.cuda.FloatTensor(img).unsqueeze(0)
    # print(img.shape)

    img = '/home/gaoxianfeng/data/face_evasion_pytorch/data/ant_aligned/source/2.JPG'
    img = cv2.imread(img)[:, :, ::-1].copy().transpose([2, 0, 1])
    img2 = torch.cuda.FloatTensor(img).unsqueeze(0)

    embedding = model(torch.cat([img1, img2], 0))

    print(torch.sum(embedding[0] * embedding[1]))
