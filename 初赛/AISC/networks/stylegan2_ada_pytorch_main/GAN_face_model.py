import os
import sys
import pickle
sys.path.insert(0,os.getcwd()+'/networks/stylegan2_ada_pytorch_main/')

def get_ffhq_face_G():
    with open('ckpts/style_GAN/ffhq.pkl','rb') as f:
        G=pickle.load(f)['G_ema'].cuda()
    return G