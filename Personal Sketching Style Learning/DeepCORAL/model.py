#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mimetypes import init
from re import T
#from typing_extensions import Self
import torch
import torch.nn as nn
from torchvision import models

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
        
class DeepCORAL(nn.Module):

	def __init__(self, num_classes=1000):
		super(DeepCORAL, self).__init__()
		self.sharedNetwork = ResNet18()
		self.fc=nn.Linear(512,num_classes)
		self.fc.apply(init_weights)

	def forward(self, source, target): # computes activations for BOTH domains
		source = self.sharedNetwork(source)
		source = self.fc(source)

		target = self.sharedNetwork(target)
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

