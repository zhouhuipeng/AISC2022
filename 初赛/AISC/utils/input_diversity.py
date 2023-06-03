import torch
import numpy as np
import scipy.stats as st
import random
from torch.nn import functional as F
from torch import nn



def input_diversity(input_tensor):
    shape=input_tensor.shape
    rnd = torch.Tensor(1, 1).uniform_(int(shape[-2]*1.1), int(shape[-2]*1.3)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).int()  # 返回一个均匀分布的矩阵

    rescaled = torch.nn.functional.interpolate(input_tensor,
                                                   size=[rnd.data.cpu().numpy()[0][0], rnd.data.cpu().numpy()[0][0]],
                                                   mode='nearest')

    h_rem = int(shape[-2]*1.3)- rnd.data.cpu().numpy()[0][0]
    w_rem = int(shape[-2]*1.3) - rnd.data.cpu().numpy()[0][0]
    pad_top = torch.Tensor(1, 1).uniform_(0, h_rem).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).int()
    pad_bottom = h_rem - pad_top.cpu().numpy()[0][0]
    pad_left = torch.Tensor(1, 1).uniform_(0, w_rem).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).int()
    pad_right = w_rem - pad_left.cpu().numpy()[0][0]  
    padded = torch.nn.functional.pad(rescaled, (pad_left.cpu().numpy()[0][0], pad_right, pad_top.cpu().numpy()[0][0], pad_bottom), "constant", value=random.randint(0,255))
    
    p=torch.Tensor(1, 1).uniform_(0, 1).data.cpu().numpy()[0][0]

    if  p <= 0.7 and p>=0.35:
        padded = torch.nn.functional.interpolate(padded,
                                                   size=[shape[-2],shape[-1]],
                                                   mode='bilinear')
    elif p<0.35 and p>=0:
        padded = torch.nn.functional.interpolate(padded,
                                                   size=[shape[-2],shape[-1]],
                                                   mode='bicubic')
    else:
        padded=input_tensor

    
    
    return padded