import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from meta_learning import MetaLearning

from omegaconf import DictConfig, OmegaConf

class ProtoNet(torch.nn.Module):
    def __init__(self, n_way, k_shot, input_dim, conv_channels, num_convolutional_blocks, convolution_size, pooling_size):
        super().__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.input_dim = input_dim
        self.conv_channels = conv_channels
        self.num_convolutional_blocks = num_convolutional_blocks
        self.conolution_size = convolution_size
        self.pooling_size = pooling_size
        self.convolutional_blocks = []
        for i in range(num_convolutional_blocks):
            conv_layer = nn.Sequential(nn.Conv2d(input_dim[-1] if i == 0 else conv_channels, 
                                                 conv_channels, kernel_size=convolution_size, padding=1),
                                                 nn.BatchNorm2d(conv_channels),
                                                 nn.ReLU(),
                                                 nn.MaxPool2d(kernel_size=pooling_size))
            self.convolutional_blocks.append(conv_layer)
        self.convolutional_blocks = nn.Sequential(*self.convolutional_blocks)

    def forward(self, s_x, s_y, q_x, q_y):
        x = torch.cat((s_x, q_x), dim=0)
        # Reshape from (28, 28, 3) --> (3, 28, 28)
        x = torch.reshape(x, (x.shape[0], x.shape[3], x.shape[1], x.shape[2]))
        x = self.convolutional_blocks(x)
        x = x.view(x.shape[0], -1)
        z_dim = x.size(-1)

        x_proto = x[:self.n_way * self.k_shot].view(self.n_way, self.k_shot, z_dim).mean(1)
        x_query = x[self.n_way * self.k_shot:]
        
        # Compute Euclidean Distance
        distances = self.euclidean_distance(x_query, x_proto)
        # Compute probabilities
        probabilities = F.log_softmax(-distances, dim=1).view(self.n_way, self.k_shot, -1)
        
        class_indices = torch.arange(0, self.n_way).view(self.n_way, 1, 1).expand(self.n_way, self.k_shot, 1).long()
        loss = -probabilities.gather(2, class_indices).squeeze().view(-1).mean()
        _, y_preds = probabilities.max(2)
        return loss, y_preds.view(-1)

    def euclidean_distance(self, x_query, x_proto):
        """Distance metric used by the paper
        """
        x_dim = x_query.shape[0]
        y_dim = x_proto.shape[0]
        z_dim = x_query.shape[1]
        
        x_query = x_query.unsqueeze(1).expand(x_dim, y_dim, z_dim)
        x_proto = x_proto.unsqueeze(0).expand(x_dim, y_dim, z_dim)
        
        return torch.pow(x_query - x_proto, 2).sum(2)