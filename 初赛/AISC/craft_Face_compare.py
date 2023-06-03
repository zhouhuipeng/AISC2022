import random as rd 
import numpy as np 
from PIL import Image
import cv2
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F 
from tqdm import tqdm
from models.StyleGAN2 import Style_GAN_ffhq

from utils.generate import read_from_yaml_file,generate_models,Attack_base_FGSM,setup_seed
from utils.dataloader import Mydataset
from utils.dct import dct_2d,idct_2d
from utils.input_diversity import input_diversity

import warnings
warnings.filterwarnings("ignore")

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def attack_dataloader(Gen,models,dataloader,hyp):
    for batch,(source_name,adv_img,target_img) in enumerate(tqdm(dataloader)):
        mask=cv2.imread('data/mask/mask.png')[:,:,::-1].astype('float').copy()
        mask/=255.0
        mask=torch.from_numpy(mask.transpose([2,0,1]).astype('int')).unsqueeze(0).to(device).float()

        attack_base_fgsm=Attack_base_FGSM(hyp)

        output_W=0
        target_img_w_plus_output=0

        ori_img=adv_img.clone()
        target_img_feature_output=[]
        for model in models:
            output=model(target_img)
            output_W+=output.clone()
            target_img_w_plus_output+=Gen.get_w_to_w_plus(output).detach().clone().to(device)
            target_img_feature_output.append(output.clone())

        noise=(target_img_w_plus_output/len(models)+Gen.get_w_to_w_plus(output_W/len(models)).detach().clone().to(device))/2
        noise.requires_grad=True
        optimer=torch.optim.Adam(params=[noise],lr=hyp['lr'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimer,T_0=5,T_mult=2)

        for iter in tqdm(range(hyp['attack_T']+150)):
            gen_face=Gen(noise).to(device)
            iter_sum_loss=0
            iter_feature_output=[]
            for model in models:
                model.zero_grad()
                output=model(input_diversity(adv_img*(-mask+1)+gen_face*mask)*2**(-rd.randint(0,2)))
                iter_feature_output.append(output)
            for i in range(len(target_img_feature_output)):
                iter_sum_loss+=(1-torch.nn.functional.cosine_similarity(target_img_feature_output[i].detach(),iter_feature_output[i]).detach())*torch.nn.functional.cosine_similarity(target_img_feature_output[i].detach(),iter_feature_output[i])

            iter_sum_loss/=len(models)
            optimer.zero_grad(set_to_none=True)
            iter_sum_loss=-iter_sum_loss
            iter_sum_loss.sum().backward()
            optimer.step()
            adv_img.data=torch.clamp(adv_img*(-mask+1)+gen_face*mask,0,255)
            scheduler.step()
                
        for iter in tqdm(range(hyp['attack_T']+150)):
            iter_sum_loss=0
            grad=torch.zeros_like(adv_img).to(device)
            for i ,model in enumerate(models):
                model.zero_grad()
                for n in range(1):
                    gauss=torch.randn(adv_img.size()[0],3,adv_img.shape[-2],adv_img.shape[-1])*(16/255)
                    gauss=gauss.cuda()
                    x_dct=dct_2d(adv_img+gauss+(hyp['ephslion']/hyp['attack_T']*1.25)*torch.sign(grad)).cuda()
                    ssa_mask=(torch.rand_like(adv_img)*2*0.5+1-0.5).cuda()
                    x_idct=idct_2d(x_dct*ssa_mask)
                    x_idct=V(x_idct,requires_grad=True)
                    x_input=attack_base_fgsm.make_input_tensor(x_idct)
                        
                    output=model(x_input)

                    iter_sum_loss=(1-torch.nn.functional.cosine_similarity(target_img_feature_output[i].detach(),output).detach())*torch.nn.functional.cosine_similarity(target_img_feature_output[i].detach(),output)

                    iter_sum_loss.sum().backward()
                    grad+=x_idct.grad
            adv_img.grad=None
            grad/=len(models)
            grad=attack_base_fgsm.getGrad(grad/len(models))
            adv_img.data=adv_img.data+(hyp['ephslion']/hyp['attack_T']*1.25)*torch.clamp(grad,-1.5,1.5)*mask

            adv_img.data=torch.clamp(adv_img.data,0,255)
            
        for i in range(len(adv_img)):
            save_img=adv_img[i].data.cpu().numpy().transpose([1,2,0]).astype('uint8')
            cv2.imwrite('res/images/{}'.format(source_name[i].split('/')[-1]),save_img[:,:,::-1])

    
if __name__=='__main__':
    setup_seed(20)
    hyp=read_from_yaml_file('hyps/AT.yaml')
    models=generate_models(hyp)

    data=Mydataset(path='data/',pairs_path='data/adv_pairs.txt')
    dataloader=torch.utils.data.DataLoader(data,batch_size=15)

    Gen=Style_GAN_ffhq(input_shape=(112,112))
    attack_dataloader(Gen,models,dataloader,hyp)