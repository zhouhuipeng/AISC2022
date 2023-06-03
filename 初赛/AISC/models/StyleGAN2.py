# https://github.com/NVlabs/stylegan2-ada-pytorch
import os 
import numpy as np
import torch

from torch import nn 
from torch.nn import functional as F 
from networks.stylegan2_ada_pytorch_main.GAN_face_model import get_ffhq_face_G

class Style_GAN_ffhq(nn.Module):
    def __init__(self,device='cuda',input_shape=(112,112)):
        super(Style_GAN_ffhq,self).__init__()
        self.model=get_ffhq_face_G()
        self.c=None
        self.model.eval().to(device)
        self.input_shape=input_shape
        self.name='Style_GAN_ffhq'
    
    def process(self,x):
        x=F.interpolate(x,self.input_shape,mode='bilinear',align_corners=True)
        x=torch.clamp(x*127.5+128,0,255)
        return x

    def get_w_to_w_plus(self,x):
        return self.model.mapping(x,None,truncation_psi=0.5,truncation_cutoff=8)

    def forward(self,x):
        out=self.model.synthesis(x,noise_mode='const',force_fp32=True)
        return self.process(out)