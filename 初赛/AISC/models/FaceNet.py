# https://github.com/timesler/facenet-pytorch
# just pip install facenet-pytorch and place the ckpts in ~/.cache/torch/checkpoint

import cv2
import torch

from facenet_pytorch import InceptionResnetV1
from torch.nn import functional as F


class FaceNet_casia(torch.nn.Module):

    def __init__(self, device='cuda', input_shape=(160,160)):
        super(FaceNet_casia, self).__init__()
        self.model = InceptionResnetV1(pretrained='casia-webface')
        self.model.to(device).eval()
        self.input_shape = input_shape
        self.name = 'FaceNet_casia'
    
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
        
class FaceNet_vggface2(torch.nn.Module):

    def __init__(self, device='cuda', input_shape=(160,160)):
        super(FaceNet_vggface2, self).__init__()
        self.model = InceptionResnetV1(pretrained='vggface2')
        self.model.to(device).eval()
        self.input_shape = input_shape
        self.name = 'FaceNet_vggface2'
    
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
    
    model = FaceNet_casia()
    img = 'data/1.JPG'
    img = cv2.imread(img)[:, :, ::-1].copy().transpose([2, 0, 1])
    img1 = torch.cuda.FloatTensor(img).unsqueeze(0)
    # print(img.shape)

    img = 'data/2.JPG'
    img = cv2.imread(img)[:, :, ::-1].copy().transpose([2, 0, 1])
    img2 = torch.cuda.FloatTensor(img).unsqueeze(0)

    embedding = model(torch.cat([img1, img2], 0))

    print(torch.sum(embedding[0] * embedding[1]))

    