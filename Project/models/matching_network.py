import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import math

class MatchingNet(nn.Module):
    def __init__(self, in_channels=1, layer_dim = 64):
        super(MatchingNetwork, self).__init__()

    self.conv_layers = []
    for i in range(4):
        if(i==0):
            input_dim = in_channels
        else:
             input_dim = layer_dim
        conv_layer = nn.Sequential(nn.Conv2d(input_dim, layer_dim, 3, 1, 1, bias=True)
                                   nn.BatchNorm2d(out_channels)
                                   nn.ReLU(True)
                                   nn.MaxPool2d(2, 2))
        self.conv_layers.append(conv_layer)
    self.conv_layers = nn.Sequential(*self.conv_layers)

    self.softmax = nn.Softmax()

    def forward(self, support_set_images, s_set_one_hots, target_image, target_label):
        encodings = []
        for i in np.arange(support_set_images.size(1)):
            x = self.conv_layers(support_set_images[:,i,:,:,:])
            x = x.view(x.size(0), -1)
            encodings.append(x)

        # produce embeddings for target images
        for i in np.arange(target_image.size(1)):
            x = self.conv_layers(target_image[:,i,:,:,:])
            x = x.view(x.size(0), -1)
            encodings.append(x)
            outputs = torch.stack(encodings)

            # distnet
            s_embeds = outputs[:-1]
            target_embed = outputs[-1]
            cos_sims = []
            for image in s_embeds:
                sum_s_embeds = np.sum(image**2, 1)
                s_mags = np.recipricol(np.sqrt(np.clip(sum_s_embeds, 1e-8, np.inf)))
                s_mags = from_numpy(s_mags)
                dot = bmm(target_embed.unsqueeze(1), image.unsqueeze(2)).squeeze()
                cos_sims.append(dot * s_mags)
            cos_sims = torch.stack(cos_sims)
            cos_sims = transpose(cos_sims, 0, 1)

            # attention
            softmax_sims = self.softmax(cos_sims)
            pred_y = bmm(softmax_sims.unsqueeze(1), s_set_one_hots).squeeze()

            # crossentropy loss
            values, indices = max(pred_y, 1)
            if i == 0:
                accuracy = torch.mean((indices.squeeze() == target_label[:,i]).float())
                loss = F.cross_entropy(pred_y, target_label[:,i].long())
            else:
                accuracy = accuracy + torch.mean((indices.squeeze() == target_label[:, i]).float())
                loss = loss + F.cross_entropy(pred_y, target_label[:, i].long())

            # delete the last target image encoding of encoded_images
            encodings.pop()

            loss = loss/target_image.size(1)
            faccuracy = accuracy/target_image.size(1)

        return loss, pred_y
