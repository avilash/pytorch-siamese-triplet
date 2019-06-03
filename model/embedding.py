import torch.nn as nn
from resnet import resnet50
from base_config import cfg

class EmbeddingResnet(nn.Module):
	def __init__(self):
		super(EmbeddingResnet, self).__init__()
		
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


class EmbeddingLeNet(nn.Module):
    def __init__(self):
        super(EmbeddingLeNet, self).__init__()
        
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 128),
                                nn.PReLU(),
                                nn.Linear(128, 64)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output