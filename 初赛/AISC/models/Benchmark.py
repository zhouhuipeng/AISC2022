#https://github.com/ShawnXYang/Face-Robustness-Benchmark

import os
import sys
# sys.path.append('networks/Face_Robustness_Benchmark_Pytorch')

import torch
from torch import nn
from torch.nn import functional as F

from networks.Benchmark.networks.IR import IR_50
import cv2





class IR50_CASIA_ArcFace_Benchmark(nn.Module):

    def __init__(self, device='cuda', input_shape=(112, 112), ckpt='ckpts/Face_Robustness_Benchmark_Pytorch/Backbone_IR_50_Epoch_16_Batch_14128_Time_2020-08-20-09-55_checkpoint.pth'):
        super(IR50_CASIA_ArcFace_Benchmark, self).__init__()
        self.model = IR_50(input_shape)
        self.model.feature = True
        self.model.load_state_dict(torch.load(ckpt,map_location=torch.device('cpu')))
        self.model.eval().to(device)
        self.input_shape = input_shape
        self.name = 'IR50_CASIA_ArcFace_Benchmark'

    def preprocess(self, x):
        # x = (x - 127.5) / 128
        # for an rgb [0, 255], any shape, (N, C, H, W) images, how to preprocess to feed into model
        assert x.size(1) == 3 or x.size(1) == 1
        if x.shape[2:] != self.input_shape:
            x = F.interpolate(x, self.input_shape, mode='bicubic', align_corners=False)

        return x

    def forward(self, x):
        # must be normalized
        out=self.model(self.preprocess(x))
        return out



class IR50_PGDSoftmax_Benchmark(nn.Module):

    def __init__(self, device='cuda', input_shape=(112, 112), ckpt='ckpts/Face_Robustness_Benchmark_Pytorch/Backbone_IR_50_Epoch_32_Batch_72352_Time_2020-08-03-20-21_checkpoint.pth'):
        super(IR50_PGDSoftmax_Benchmark, self).__init__()
        self.model = IR_50(input_shape)
        self.model.feature = True
        self.model.load_state_dict(torch.load(ckpt,map_location=torch.device('cpu')))
        self.model.eval().to(device)
        self.input_shape = input_shape
        self.name = 'IR50_PGDSoftmax_Benchmark'

    def preprocess(self, x):
        # x = (x - 127.5) / 128
        # for an rgb [0, 255], any shape, (N, C, H, W) images, how to preprocess to feed into model
        assert x.size(1) == 3 or x.size(1) == 1
        if x.shape[2:] != self.input_shape:
            x = F.interpolate(x, self.input_shape, mode='bicubic', align_corners=False)

        return x

    def forward(self, x):
        # must be normalized
        out=self.model(self.preprocess(x))
        return out

class IR50_PGDCosFace_Benchmark(nn.Module):

    def __init__(self, device='cuda', input_shape=(112, 112), ckpt='ckpts/Face_Robustness_Benchmark_Pytorch/Backbone_IR_50_Epoch_32_Batch_60288_Time_2020-08-04-03-37_checkpoint.pth'):
        super(IR50_PGDCosFace_Benchmark, self).__init__()
        self.model = IR_50(input_shape)
        self.model.feature = True
        self.model.load_state_dict(torch.load(ckpt,map_location=torch.device('cpu')))
        self.model.eval().to(device)
        self.input_shape = input_shape
        self.name = 'IR50_PGDCosFace_Benchmark'

    def preprocess(self, x):
        # x = (x - 127.5) / 128
        # for an rgb [0, 255], any shape, (N, C, H, W) images, how to preprocess to feed into model
        assert x.size(1) == 3 or x.size(1) == 1
        if x.shape[2:] != self.input_shape:
            x = F.interpolate(x, self.input_shape, mode='bicubic', align_corners=False)

        return x

    def forward(self, x):
        # must be normalized
        out=self.model(self.preprocess(x))
        return out

class IR50_PGDArcFace_Benchmark(nn.Module):

    def __init__(self, device='cuda', input_shape=(112, 112), ckpt='ckpts/Face_Robustness_Benchmark_Pytorch/Backbone_IR_50_Epoch_16_Batch_30144_Time_2020-08-20-06-41_checkpoint.pth'):
        super(IR50_PGDArcFace_Benchmark, self).__init__()
        self.model = IR_50(input_shape)
        self.model.feature = True
        self.model.load_state_dict(torch.load(ckpt,map_location=torch.device('cpu')))
        self.model.eval().to(device)
        self.input_shape = input_shape
        self.name = 'IR50_PGDArcFace_Benchmark'

    def preprocess(self, x):
        # x = (x - 127.5) / 128
        # for an rgb [0, 255], any shape, (N, C, H, W) images, how to preprocess to feed into model
        assert x.size(1) == 3 or x.size(1) == 1
        if x.shape[2:] != self.input_shape:
            x = F.interpolate(x, self.input_shape, mode='bicubic', align_corners=False)

        return x

    def forward(self, x):
        # must be normalized
        out=self.model(self.preprocess(x))
        return out


class IR50_TradesCosFace_Benchmark(nn.Module):

    def __init__(self, device='cuda', input_shape=(112, 112), ckpt='ckpts/Face_Robustness_Benchmark_Pytorch/Backbone_IR_50_Epoch_32_Batch_60288_Time_2020-09-06-04-48_checkpoint.pth'):
        super(IR50_TradesCosFace_Benchmark, self).__init__()
        self.model = IR_50(input_shape)
        self.model.feature = True
        self.model.load_state_dict(torch.load(ckpt,map_location=torch.device('cpu')))
        self.model.eval().to(device)
        self.input_shape = input_shape
        self.name = 'IR50_TradesCosFace_Benchmark'

    def preprocess(self, x):
        # x = (x - 127.5) / 128
        # for an rgb [0, 255], any shape, (N, C, H, W) images, how to preprocess to feed into model
        assert x.size(1) == 3 or x.size(1) == 1
        if x.shape[2:] != self.input_shape:
            x = F.interpolate(x, self.input_shape, mode='bicubic', align_corners=False)

        return x

    def forward(self, x):
        # must be normalized
        out=self.model(self.preprocess(x))
        return out




if __name__=='__main__':
    img = torch.randn(1, 3, 112, 96).cuda()
    net=CosFace_Benchmark()
    print(net(img))