#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision import models

class DDCNet(nn.Module):
	"""
	Deep domain confusion network as defined in the paper:
	https://arxiv.org/abs/1412.3474
    :param num_classes: int --> office dataset has 31 different classes
	"""
	def __init__(self, num_classes=1000):
		super(DDCNet, self).__init__()
		self.sharedNetwork = ResNet18()

		self.bottleneck = nn.Sequential(
			nn.Linear(512, 512),
			nn.ReLU(inplace=True)
		)

		self.fc = nn.Sequential(
            nn.Linear(512,num_classes)      
		)

	def forward(self, source, target): # computes activations for BOTH domains
		source = self.sharedNetwork(source)
		source = self.bottleneck(source)
		source = self.fc(source)

		target = self.sharedNetwork(target)
		target = self.bottleneck(target)
		target = self.fc(target)

		return source, target

class ResNet18(nn.Module):
    def __init__(self, class_num=1000):
        super(ResNet18, self).__init__()
        model_resnet = models.resnet18(pretrained=False)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = nn.AdaptiveMaxPool2d((1,1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        return x

