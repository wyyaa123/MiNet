import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision
from torchvision import transforms
import torch.utils.data as data
#import torchsnooper
import cv2

BATCH_SIZE = 128
EPOCH = 10

# BRelu used for GPU. Need to add that reference in pytorch source file.
class BRelu(nn.Hardtanh):
	def __init__(self, inplace=False):
		super(BRelu, self).__init__(0., 1., inplace)
		
	def extra_repr(self):
		inplace_str = 'inplace=True' if self.inplace else ''
		return inplace_str


class DehazeNet(nn.Module):
	def __init__(self, input=16, groups=4):
		super(DehazeNet, self).__init__()
		self.input = input
		self.groups = groups
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.input, kernel_size=5)
		self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, padding=2)
		self.conv4 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=7, padding=3)
		self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1)
		self.conv5 = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=6)
		self.brelu = BRelu()
		for name, m in self.named_modules():
			if isinstance(m, nn.Conv2d):
				# 初始化 weight 和 bias
				nn.init.normal_(m.weight, mean=0,std=0.001)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
	
	def Maxout(self, x, groups):
		x = x.reshape(x.shape[0], groups, x.shape[1]//groups, x.shape[2], x.shape[3])
		x, y = torch.max(x, dim=2, keepdim=True)
		out = x.reshape(x.shape[0],-1, x.shape[3], x.shape[4])
		return out
	#BRelu used to CPU. It can't work on GPU.
	def BRelu(self, x):
		x = torch.max(x, torch.zeros(x.shape[0],x.shape[1],x.shape[2],x.shape[3]))
		x = torch.min(x, torch.ones(x.shape[0],x.shape[1],x.shape[2],x.shape[3]))
		return x
	
	def forward(self, x):
		out = self.conv1(x)
		out = self.Maxout(out, self.groups)
		out1 = self.conv2(out)
		out2 = self.conv3(out)
		out3 = self.conv4(out)
		y = torch.cat((out1,out2,out3), dim=1)
		#print(y.shape[0],y.shape[1],y.shape[2],y.shape[3],)
		y = self.maxpool(y)
		#print(y.shape[0],y.shape[1],y.shape[2],y.shape[3],)
		y = self.conv5(y)
		# y = self.relu(y)
		# y = self.BRelu(y)
		#y = torch.min(y, torch.ones(y.shape[0],y.shape[1],y.shape[2],y.shape[3]))
		y = self.brelu(y)
		y = y.reshape(y.shape[0],-1)
		return y
	
if __name__ == "__main__":

    net = DehazeNet()
	
    print(net.__name__)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    print(macs, params)

