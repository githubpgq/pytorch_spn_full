import torch
import torch.nn as nn
from torchvision.models import vgg16
from torch.autograd import Variable
from collections import OrderedDict
from lib.pytorch_spn.modules.gaterecurrent2dnoind import GateRecurrent2dnoind

"""
Embeded gate for SPN for multi-structure data segmentation
"""

class LayerNorm(nn.Module):

	def __init__(self, eps=1e-5):
		super().__init__()
		self.register_parameter('gamma', None)
		self.register_parameter('beta', None)
		self.eps = eps

	def forward(self, x):
		if self.gamma is None:
			self.gamma = nn.Parameter(torch.ones(x.size()).cuda())
		if self.beta is None:
			self.beta = nn.Parameter(torch.zeros(x.size()).cuda())
		mean = x.mean(1, keepdim=True).repeat(1,x.size()[1],1,1)
		std = x.std(1, keepdim=True).repeat(1,x.size()[1],1,1)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Embeded_gate(nn.Module):
	"""
	Learable bias for Euid based similarity.
	"""
	def __init__(self, horizontal, reverse, mode = "L2"):
		super(Embeded_gate, self).__init__()
		if mode in "dot":
			self.norm = LayerNorm()
		else:
			self.register_parameter('bias', None)
		self.horizontal = horizontal
		self.reverse = reverse
		self.mode = mode
		# if self.horizontal:
		if not self.reverse:
			self.pad_1 = nn.ZeroPad2d((1,0,1,0))
			self.pad_2 = nn.ZeroPad2d((1,0,0,0))
			self.pad_3 = nn.ZeroPad2d((1,0,0,1))
		else:
			self.pad_1 = nn.ZeroPad2d((0,1,1,0))
			self.pad_2 = nn.ZeroPad2d((0,1,0,0))
			self.pad_3 = nn.ZeroPad2d((0,1,0,1))
		# else:

	def forward(self, x):
		n,c,h,w = x.size()
		if self.mode not in "dot" and self.bias is None:
			self.bias = nn.Parameter(-0.5*torch.ones(1,c,1,1).cuda())

		if self.mode in "dot":
			if self.horizontal:
				x_norm = self.norm(x)
			else:
				x_norm = self.norm(x).transpose(3,2)
		else:
			if self.horizontal:
				x_norm = x
			else:
				x_norm = x.transpose(3,2)

		if not self.reverse:
			if self.mode in "dot":
				G1 = self.pad_1(torch.sum(torch.mul(x_norm[:,:,:-1,:-1], x_norm[:,:,1:,1:]), 1, keepdim=True).repeat(1,c,1,1))
				G2 = self.pad_2(torch.sum(torch.mul(x_norm[:,:,:,:-1], x_norm[:,:,:,1:]), 1, keepdim=True).repeat(1,c,1,1))
				G3 = self.pad_3(torch.sum(torch.mul(x_norm[:,:,1:,:-1], x_norm[:,:,:-1,1:]), 1, keepdim=True).repeat(1,c,1,1))
			elif self.mode in "meanL2":
				G1 = self.pad_1(torch.sum(torch.exp(-torch.pow((x_norm[:,:,:-1,:-1] - x_norm[:,:,1:,1:]), 2)), 1, keepdim=True).repeat(1,c,1,1)) + self.bias.repeat(n,1,h,w)
				G2 = self.pad_2(torch.sum(torch.exp(-torch.pow((x_norm[:,:,:,:-1] - x_norm[:,:,:,1:]), 2)), 1, keepdim=True).repeat(1,c,1,1)) + self.bias.repeat(n,1,h,w)
				G3 = self.pad_3(torch.sum(torch.exp(-torch.pow((x_norm[:,:,1:,:-1] - x_norm[:,:,:-1,1:]), 2)), 1, keepdim=True).repeat(1,c,1,1)) + self.bias.repeat(n,1,h,w)
			elif self.mode in "L2":
				G1 = self.pad_1(torch.exp(-torch.pow((x_norm[:,:,:-1,:-1] - x_norm[:,:,1:,1:]), 2))) + self.bias.repeat(n,1,h,w)
				G2 = self.pad_2(torch.exp(-torch.pow((x_norm[:,:,:,:-1] - x_norm[:,:,:,1:]), 2))) + self.bias.repeat(n,1,h,w)
				G3 = self.pad_3(torch.exp(-torch.pow((x_norm[:,:,1:,:-1] - x_norm[:,:,:-1,1:]), 2))) + self.bias.repeat(n,1,h,w)
			else:
				raise Exception("unknow distance matrix (not in: dot, meanL2, L2).")
		else:
			if self.mode in "dot":
				G1 = self.pad_1(torch.sum(torch.mul(x_norm[:,:,:-1,1:], x_norm[:,:,1:,:-1]), 1, keepdim=True).repeat(1,c,1,1))
				G2 = self.pad_2(torch.sum(torch.mul(x_norm[:,:,:,1:], x_norm[:,:,:,:-1]), 1, keepdim=True).repeat(1,c,1,1))
				G3 = self.pad_3(torch.sum(torch.mul(x_norm[:,:,1:,1:], x_norm[:,:,:-1,:-1]), 1, keepdim=True).repeat(1,c,1,1))
			elif self.mode in "meanL2":
				G1 = self.pad_1(torch.sum(torch.exp(-torch.pow((x_norm[:,:,:-1,1:], - x_norm[:,:,1:,:-1]), 2)), 1, keepdim=True).repeat(1,c,1,1)) + self.bias.repeat(n,1,h,w)
				G2 = self.pad_2(torch.sum(torch.exp(-torch.pow((x_norm[:,:,:,1:], - x_norm[:,:,:,:-1]), 2)), 1, keepdim=True).repeat(1,c,1,1)) + self.bias.repeat(n,1,h,w)
				G3 = self.pad_3(torch.sum(torch.exp(-torch.pow((x_norm[:,:,1:,1:] - x_norm[:,:,:-1,:-1]), 2)), 1, keepdim=True).repeat(1,c,1,1)) + self.bias.repeat(n,1,h,w)
			elif self.mode in "L2":
				G1 = self.pad_1(torch.exp(-torch.pow((x_norm[:,:,:-1,1:], - x_norm[:,:,1:,:-1]), 2))) + self.bias.repeat(n,1,h,w)
				G2 = self.pad_2(torch.exp(-torch.pow((x_norm[:,:,:,1:], - x_norm[:,:,:,:-1]), 2))) + self.bias.repeat(n,1,h,w)
				G3 = self.pad_3(torch.exp(-torch.pow((x_norm[:,:,1:,1:] - x_norm[:,:,:-1,:-1]), 2))) + self.bias.repeat(n,1,h,w)
			else:
				raise Exception("unknow distance matrix (not in: dot, meanL2, L2).")

		if self.horizontal:
			return G1, G2, G3
		else:
			return G1.transpose(3,2), G2.transpose(3,2), G3.transpose(3,2)


class spn_block(nn.Module):
	def __init__(self, horizontal, reverse, mode = "dot"):
		super(spn_block, self).__init__()
		self.embeded_gate = Embeded_gate(horizontal, reverse, mode)
		self.propagator = GateRecurrent2dnoind(horizontal,reverse)

	def forward(self, x, z):
		G1, G2, G3 = self.embeded_gate(z)
		sum_abs = G1.abs() + G2.abs() + G3.abs()
		sum_abs.data[sum_abs.data == 0] = 1e-6
		mask_need_norm = sum_abs.ge(1)
		mask_need_norm = mask_need_norm.float()
		G1_norm = torch.div(G1, sum_abs)
		G2_norm = torch.div(G2, sum_abs)
		G3_norm = torch.div(G3, sum_abs)

		G1 = torch.add(-mask_need_norm, 1) * G1 + mask_need_norm * G1_norm
		G2 = torch.add(-mask_need_norm, 1) * G2 + mask_need_norm * G2_norm
		G3 = torch.add(-mask_need_norm, 1) * G3 + mask_need_norm * G3_norm

		return self.propagator(x,G1,G2,G3)

def fixLayer(layer):
	for param in layer.parameters():
		param.requires_grad = False

class BaseModule(nn.Module):
	def __init__(self, base, fix=0):
		super(BaseModule,self).__init__()
		self.layer0 = base._modules['0']
		self.layer1 = base._modules['1']
		self.layer2 = base._modules['2']
		self.layer3 = base._modules['3']
		self.layer4 = base._modules['4']
		self.layer5 = base._modules['5']
		self.layer6 = base._modules['6']
		self.layer7 = base._modules['7']
		self.layer8 = base._modules['8']

		if(fix == 1):
			print('fixing base networks...`')

			fixLayer(self.layer0)

			fixLayer(self.layer1)

			fixLayer(self.layer2)

			fixLayer(self.layer3)

			fixLayer(self.layer4)

			fixLayer(self.layer5)

			fixLayer(self.layer6)

			fixLayer(self.layer7)

			fixLayer(self.layer8)

		self.convs0 = nn.Sequential(nn.Conv2d(16,32,3,2,1),

									nn.ReLU(inplace=True),

									nn.Conv2d(32,64,3,2,1))

		self.convs1 = nn.Sequential(nn.Conv2d(16,32,3,2,1),

									nn.ReLU(inplace=True),

									nn.Conv2d(32,64,3,2,1))

		self.convs2 = nn.Conv2d(32,64,3,2,1)

		self.cnn = nn.Sequential(nn.Conv2d(64*4,256,3,1,1),

								 nn.ReLU(inplace=True),

								 nn.Conv2d(256,256,3,2,1),

								 nn.ReLU(inplace=True),

								 nn.Conv2d(256,512,3,1,1))

		# spn blocks
		self.left_right = spn_block(True,False)
		self.right_left = spn_block(True,True)
		self.top_down = spn_block(False, False)
		self.down_top = spn_block(False,True)


	def forward(self,x):
		output = dict()
		output['0'] = self.layer0(x)
		output['1'] = self.layer1(output['0'])
		output['2'] = self.layer2(output['1'])
		output['3'] = self.layer3(output['2'])
		output['4'] = self.layer4(output['3'])
		output['5'] = self.layer5(output['4'])
		output['6'] = self.layer6(output['5'])
		output['7'] = self.layer7(output['6'])
		output['8'] = self.layer8(output['7'])

		out0 = self.convs0(output['0'])

		out1 = self.convs1(output['1'])

		out2 = self.convs2(output['2'])

		cnn_in = torch.cat((out0,out1,out2,output['3']),dim=1)

		cnn_out = self.cnn(cnn_in)

		out1 = self.left_right(output['8'], cnn_out)
		out2 = self.right_left(output['8'], cnn_out)
		out3 = self.top_down(output['8'], cnn_out)
		out4 = self.down_top(output['8'], cnn_out)
		out = torch.max(out1,out2)
		out = torch.max(out,out3)
		output['8'] = torch.max(out,out4)

		return output
