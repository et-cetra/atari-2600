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
        self.conv.apply(self.init_weights)

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.fc.apply(self.init_weights)

    # Init the values of the neural network. We use Kaiming (He)
    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            m.bias.data.fill_(0)

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

class DuelingDQN(nn.Module):
    '''
    Extends the vanilla network by processing the output of 
     the convolutional layer through two independent paths.
     One path estimates the value of state, and one estimates
     the value of action.

    Class references code from Lapan, M. <https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter07/06_dqn_dueling.py>
    '''
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv.apply(self.init_weights)

        # Advantage of action stream of the neural net
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        self.fc_advantage.apply(self.init_weights)

        # Value of state stream of the neural net
        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.fc_value.apply(self.init_weights)

    def init_weights(self, m):
        # Init the values of the neural network. We use Kaiming (He)
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            m.bias.data.fill_(0)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        floatX = x.float()
        conv_out = self.conv(floatX).view(floatX.size()[0], -1)
        val = self.fc_value(conv_out)
        adv = self.fc_advantage(conv_out)
        return val + (adv - adv.mean(dim=1, keepdim=True))