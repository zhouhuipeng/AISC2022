import os
import sys
sys.path.insert(0, './')
import torch
import cv2
import numpy as np
import argparse
import net
from tqdm import tqdm
from models.CosFace import CosFace
batch_size = 50

def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i*n/n_folds):int((i+1)*n/n_folds)]
        train = list(set(base)-set(test))
        folds.append([train,test])
    return folds

def eval_acc(threshold, diff, gts):
    assert len(diff) == len(gts)
    return np.sum((diff < threshold) == gts) / len(diff)

def find_best_threshold(thresholds, predicts, gts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts, gts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

config = {}
def read_pairs(path):
	with open(os.path.join('/home/nas928/dataset/lfw/', 'pairs.txt')) as f:
		lines = f.readlines()
	suffix = '.jpg'
	pairs = []
	for line in lines:
		line = line.strip('\n').split('\t')
		if len(line) == 3:
			pair = []
			pair.append(os.path.join(line[0], line[0] + '_' + line[1].zfill(4) + suffix))
			pair.append(os.path.join(line[0], line[0] + '_' + line[2].zfill(4) + suffix))
			pair.append(0)
			pairs.append(pair)
		elif len(line) == 4:
			pair = []
			pair.append(os.path.join(line[0], line[0] + '_' + line[1].zfill(4) + suffix))
			pair.append(os.path.join(line[2], line[2] + '_' + line[3].zfill(4) + suffix))
			pair.append(1)
			pairs.append(pair)
		else:
			pass
	return pairs


def test(data_path, pairs, model, shape, model_name):
	num = len(pairs)
	result_cos = np.empty(shape=(num))
	gt = np.empty(shape=(num))

	zero_matrix = np.zeros(shape=(1, shape[0], shape[1], 3))
	_pairs = [[], []]
	for i in tqdm(range(num)):
		
		for iii, filename in enumerate(pairs[i][:2]):
			image = cv2.imread(os.path.join(data_path, filename))[:, :, ::-1].copy().transpose([2, 0, 1])
			_pairs[iii].append(image)
		if i % batch_size == batch_size - 1:
			with torch.no_grad():
				_pairs[0] = model.forward(torch.cuda.FloatTensor(np.array(_pairs[0])))
				_pairs[1] = model.forward(torch.cuda.FloatTensor(np.array(_pairs[1])))
			result_cos[i - batch_size + 1 : i + 1] = torch.sum(_pairs[0] * _pairs[1], dim=1).cpu().numpy()
			_pairs = [[], []]
		gt[i] = pairs[i][2]
	assert _pairs == [[], []]

	# from matplotlib import pyplot as plt
	# import matplotlib
	# matplotlib.use('agg')
	# plt.scatter(list(range(len(result_cos))), result_cos)
	# plt.savefig('dis.png')

	accuracy = []
	thd = []
	folds = KFold(n=6000, n_folds=10, shuffle=False)
	# thresholds = np.arange(-1.0, 1.0, 0.005)
	thresholds = sorted(np.unique(result_cos))
	predicts = np.array(result_cos)
	gts = np.array(gt)
	for idx, (train, test) in enumerate(folds):
		best_thresh = find_best_threshold(thresholds, predicts[train], gts[train])
		accuracy.append(eval_acc(best_thresh, predicts[test], gts[test]))
		thd.append(best_thresh)
	print('{} LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(model_name, np.mean(accuracy), np.std(accuracy), np.mean(thd)))
	config[model_name] = {
		'threshold': np.mean(thd),
		'cos_acc': np.mean(accuracy),
		'distribute': (np.min(result_cos), np.max(result_cos))
	}
def callmain(ckpt):
	dst = "/home/nas928/dataset/lfw/lfw-112X96/"
	pairs = read_pairs(dst)
	# ckpt="/home/sshfs928/zhp/FaceModelZOO_pytorch-master-final/Fusing_Attack/networks/CosFace_pytorch/checkpoint/CosFace_0_checkpoint.pth"
	# ckpt="/home/sshfs928/zhp/FaceModelZOO_pytorch-master-final/Fusing_Attack/ckpts/CosFace_pytorch/ACC99.28.pth"
	model=CosFace(ckpt=ckpt).to('cuda')
	
	test(dst, pairs, model, shape=(112,96),model_name="CosFace")
	for k, v in config.items():
		print('\'{}\':{},'.format(k, v))

if __name__=='__main__':
	callmain()
	
