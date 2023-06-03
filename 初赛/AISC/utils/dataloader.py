import cv2
from PIL import Image
import torch
import torchvision
import numpy as np
from torch.utils.data.dataset import Dataset

class Mydataset(Dataset):
    def __init__(self,path,pairs_path):
        super(Mydataset,self).__init__()
        self.path=path
        self.source_path=[]
        self.target_path=[]
        with open(pairs_path,'r') as f:
            for line in f.readlines():
                line=line.strip()
                source,target=line.split(' ')[0],line.split(' ')[1]
                self.source_path.append(source)
                self.target_path.append(target)
    
    def process_img(self,img_path):
        img=cv2.imread(img_path)[:,:,::-1].astype('float').copy()
        img=torch.from_numpy(img.transpose([2,0,1])).to('cuda').float()
        return img
    
    def __len__(self):
        return len(self.target_path)

    def __getitem__(self,index):
        source_name=self.source_path[index]
        target_name=self.target_path[index]

        cv2.setNumThreads(0)
        source_img_tensor=self.process_img(self.path+source_name)
        target_img_tensor=self.process_img(self.path+target_name)

        return source_name,source_img_tensor,target_img_tensor
