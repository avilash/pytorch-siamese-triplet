import torch.nn as nn
from resnet import resnet50
from base_config import cfg

class Embedding(nn.Module):
	def __init__(self):
		super(Embedding, self).__init__()
		resnet = resnet50(pretrained=True)
		self.model = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu, resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3,resnet.layer4)
		# Fix blocks
		for p in self.model[0].parameters(): p.requires_grad=False
		for p in self.model[1].parameters(): p.requires_grad=False
		if cfg.RESNET.FIXED_BLOCKS >= 3:
			for p in self.model[6].parameters(): p.requires_grad=False
		if cfg.RESNET.FIXED_BLOCKS >= 2:
			for p in self.model[5].parameters(): p.requires_grad=False
		if cfg.RESNET.FIXED_BLOCKS >= 1:
			for p in self.model[4].parameters(): p.requires_grad=False

		def set_bn_fix(m):
			classname = m.__class__.__name__
			if classname.find('BatchNorm') != -1:
				for p in m.parameters(): p.requires_grad=False

		self.model.apply(set_bn_fix)

	def forward(self, x):
	    return self.model.forward(x)