#
# Code assembled and tweaked by:
#   Joshua Derbe
#   Sean Luo
#   Mariya Shmalko
#   Arkie Owen
#
#
# Base algorithm outline created with heavy reference to
#   Lapan, M. (2018) Deep Reinforcement Learning Hands-On, Packt Publishing.
#

import numpy as np
import torch
import torch.nn as nn


class VanillaDQN(nn.Module):
    '''
    This is a simple DQN with first a convolution network
     to extract features from the input frame. Then it uses 
     a fully connected network to train on the extract 
     features to produce action scores.
    
    Class references code from Lapan, M. <https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/dqn_model.py>
    '''
    def __init__(self, input_shape, n_actions:int):
        # input_shape[0] is the n.o of channels of the input image
        # i.e. in RGB images, it is 3
        # input_shape is of the form [Channel,Height,Width]
        # n_actions: the number of possible actions in action space.
        super(VanillaDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    '''
    act as bridge between convolution and fc by making
     the convolution's ouput shape dynamic to input
    '''
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    # forward pass of network
    def forward(self, x):
        fx = x.float()
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)