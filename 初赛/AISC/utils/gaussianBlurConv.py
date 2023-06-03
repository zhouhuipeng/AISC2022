import torch
import numpy as np
import scipy.stats as st

from torch.nn import functional as F
from torch import nn

class GaussianBlurConv(torch.nn.Module):
	def __init__(self, kernellen=3, sigma=4, channels=3, device='cuda'):
		super(GaussianBlurConv, self).__init__()
		if kernellen % 2 == 0:
			kernellen += 1
		self.kernellen = kernellen
		kernel = self.gkern(kernellen, sigma).astype(np.float32)
		self.channels = channels
		kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
		kernel = np.repeat(kernel, self.channels, axis=0)
		self.weight = torch.nn.Parameter(data=kernel, requires_grad=False).to(device)

	def forward(self, x):
		padv = (self.kernellen - 1) // 2
		x = nn.functional.pad(x, pad=(padv, padv, padv, padv), mode='replicate')
		x = F.conv2d(x, self.weight, stride=1, padding=0, groups=self.channels)
		return x

	def gkern(self, kernlen=3, nsig=4):
		"""Returns a 2D Gaussian kernel array."""
		interval = (2*nsig+1.)/(kernlen)
		x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
		kern1d = np.diff(st.norm.cdf(x))
		kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
		kernel = kernel_raw/kernel_raw.sum()
		return kernel