import torch
import cv2
from networks.Face_Robustness_Benchmark_Pytorch.networks.ResNet import ResNet_50
from networks.Face_Robustness_Benchmark_Pytorch.networks.Mobilenet import MobileNet
from networks.Face_Robustness_Benchmark_Pytorch.networks.ShuffleNet import ShuffleNet
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

Net = ResNet_50((112,112))
Net.load_state_dict(torch.load("../../../ckpts/Face_Robustness_Benchmark_Pytorch/Backbone_ResNet_50_Epoch_36_Batch_204696_Time_2019-04-14-14-44_checkpoint.pth"))
Net.to(device).eval()
print(Net)
img = '../../../data/source.png'
img = cv2.imread(img)[:, :, ::-1].copy().transpose([2, 0, 1])
img1 = torch.cuda.FloatTensor(img).unsqueeze(0)
img_cropped = torch.nn.functional.interpolate(img1,
                                                      size=[112, 112],
                                                      mode='bilinear')
print(Net(img_cropped).shape)

# Backbone_Mobilenet_Epoch_125_Batch_710750_Time_2019-04-14-18-15_checkpoint.pth
net=MobileNet(2)
net.load_state_dict(torch.load("../../../ckpts/Face_Robustness_Benchmark_Pytorch/Backbone_Mobilenet_Epoch_125_Batch_710750_Time_2019-04-14-18-15_checkpoint.pth"))
net.to(device).eval()
print(net)
print(net(img_cropped).shape)

net=ShuffleNet(pooling='GDConv')
net.load_state_dict(torch.load("../../../ckpts/Face_Robustness_Benchmark_Pytorch/Backbone_ShuffleNet_Epoch_124_Batch_1410128_Time_2019-05-05-02-33_checkpoint.pth"))
net.to(device).eval()
print(net)
print(net(img_cropped).shape)