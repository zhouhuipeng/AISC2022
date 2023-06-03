import yaml
import torch
import random 
import numpy as np

from models.ArcFace import ArcFace
from models.CosFace import CosFace
from models.FaceNet import FaceNet_casia,FaceNet_vggface2
from models.ResNet import Resnet50
from models.MobileNet import MobileNetV2
from models.InsightFace_Pytorch import MobileFacenet
from models.Benchmark import IR50_CASIA_ArcFace_Benchmark,IR50_PGDSoftmax_Benchmark,IR50_PGDCosFace_Benchmark,IR50_PGDArcFace_Benchmark,IR50_TradesCosFace_Benchmark

from utils.gaussianBlurConv import GaussianBlurConv
from utils.input_diversity import input_diversity

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def read_from_yaml_file(yaml_path):
    assert yaml_path.endswith(".yaml")
    f=open(yaml_path,'r',encoding='utf-8')
    cfg=f.read()
    hyp=yaml.load(cfg,Loader=yaml.FullLoader)
    return hyp

def generate_model_dict():
    model_list={
        'ArcFace':ArcFace,
        'CosFace':CosFace,
        'FaceNet_casia':FaceNet_casia,
        'FaceNet_vggface2':FaceNet_vggface2,
        'Resnet50':Resnet50,
        'MobileNetV2':MobileNetV2,
        'MobileFacenet':MobileFacenet,
        'IR50_CASIA_ArcFace_Benchmark':IR50_CASIA_ArcFace_Benchmark,
        'IR50_PGDSoftmax_Benchmark':IR50_PGDSoftmax_Benchmark,
        'IR50_PGDCosFace_Benchmark':IR50_PGDCosFace_Benchmark,
        'IR50_PGDArcFace_Benchmark':IR50_PGDArcFace_Benchmark,
        'IR50_TradesCosFace_Benchmark':IR50_TradesCosFace_Benchmark,
    }
    return model_list


def generate_models(hyp):
    model_list=generate_model_dict()
    models=[]
    for modelname in hyp['sourceModels']:
        models.append(model_list[modelname]())
    return models


class Attack_base_FGSM():
    def __init__(self,hyp):
        self.hyp=hyp
        self.sum_grad=torch.zeros(1).cuda()
        self.grad=None
        self.GaussianSmooth=GaussianBlurConv(kernellen=5).cuda()
        self.alpha=self.hyp['ephslion']/self.hyp['attack_T']*2
    
    def getGrad(self,grad):
        self.grad=grad
        if 'TIM' in self.hyp['attackMethod']:
            self.grad=self.GaussianSmooth(self.grad)
        if 'MIM' in self.hyp['attackMethod']:
            self.grad=self.grad/self.grad.abs().mean(dim=[1,2,3],keepdim=True)
            self.sum_grad=self.grad+self.sum_grad
            self.sum_grad=self.sum_grad/self.sum_grad.abs().mean(dim=[1,2,3],keepdim=True)
            self.grad=self.sum_grad
        return self.grad
    
    def make_input_tensor(self,img):
        operater=[]
        if 'DIM' in self.hyp['attackMethod']:
            operater.append('DIM')
        if 'SIM' in self.hyp['attackMethod']:
            operater.append('DIM')
        if 'NIM' in self.hyp['attackMethod']:
            operater.append('DIM')

        rule_list=[x for x in range(len(operater))]
        random.shuffle(rule_list)
        rule={}
        for index,item in enumerate(operater):
            rule.setdefault(item,rule_list[index])
        operater=sorted(operater,key=lambda x:rule[x])

        for name in operater:
            if 'DIM' in name:
                img=input_diversity(img)
            if 'SIM' in name:
                j=random.randint(0,2)
                img=img*2**(-j)
            if 'NIM' in name:
                img=img+self.alpha*torch.sign(self.sum_grad)
        return img