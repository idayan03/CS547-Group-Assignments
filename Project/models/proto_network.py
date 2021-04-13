import os
import time
import torch
import numpy as np
import multiprocessing as mp
from torch.autograd import Variable
from meta_learning import MetaLearning

from omegaconf import DictConfig, OmegaConf


class ProtoNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()


        self.network = torch.nn.Sequential(
            self.convolution_block(input_dim[2], hidden_dim),
            self.convolution_block(hidden_dim, hidden_dim),
            self.convolution_block(hidden_dim, hidden_dim),
            self.convolution_block(hidden_dim, output_dim)
        )

    def forward(self, s_x, s_y, q_x, q_y):

        return

    def convolution_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )