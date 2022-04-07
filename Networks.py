import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import os
import matplotlib
import copy
import tensorflow as tf


class DiscNet(nn.Module):
    def __init__(self, width = 0.01, norm = 'batch'):
        super(DiscNet, self).__init__()

        self.hidden_layers = 2

        # define network topology
        conv1 = tf.keras.layers.Conv2D(1, 32, kernel_size=5, padding = 2, stride = 2, bias=True)
        norm1 = self.normalization(32,2,norm)
        conv2 = tf.keras.layers.Conv2D(32, 64, kernel_size=4, padding = 2, stride =2, bias=True)
        norm2 = self.normalization(64,2,norm)
        linear = tf.keras.layers.Dense(64*64,10, bias=True)
        norm3 = self.normalization(10,1,norm)

        # pytorch ModuleDict https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html
        self.layers = nn.ModuleDict({'cv1': conv1, 'n1': norm1, 'cv2': conv2, 'n2': norm2, 'fc':linear, 'n3':norm3})

        # initialize weights
        for layer in self.layers.keys():
            if not(norm in layer):
                # from pytorch init documentation https://pytorch.org/docs/stable/nn.init.html
                nn.init.normal_(self.layers[layer].weight, mean=0, std=width)
          
    def normalization(self, size, dim, norm):
        # from pytorch documentation: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        if norm == 'batch':
            if dim==2:
                return nn.BatchNorm2d(size, affine=True, track_running_stats=True)
            else:
                return nn.BatchNorm1d(size, affine=True, track_running_stats=True)
        else:
            if dim==2:
                return nn.InstanceNorm2d(size, affine=False, track_running_stats=False)
            else:
                return nn.InstanceNorm1d(size, affine=False, track_running_stats=False)

    def forward(self, x):
        x = self.layers['cv1'](x)
        x = self.layers['n1'](x)
        x = Binarization.apply(x)

        x = self.layers['cv2'](x)
        x = self.layers['n2'](x)
        x = Binarization.apply(x)

        # flatten x before the fully connected layer via pytorch's view function
        x = x.view(x.size(0), -1)
        x = self.layers['fc'](x)
        x = self.layers['n3'](x)

        return x
        
    def save_nn_states(self):
        nn_states = []
        if 'n1' in self.layers.keys():
            for l in range(self.hidden_layers+1):
                bn = copy.deepcopy(self.layers['nn'+str(l+1)].state_dict())
                bn_states.append(bn)
        return bn_states

    def load_nn_states(self, bn_states):
        if 'n1' in self.layers.keys():
            for l in range(self.hidden_layers+1):
                self.layers['nn'+str(l+1)].load_state_dict(bn_states[l])