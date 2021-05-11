import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, conv_channels, embed_dim,
                 kernel_size=5, num_conv_layers=2, num_pair_layers=2,
                 wh_after_conv=4):
        super(SiameseNetwork, self).__init__()

        self.conv_layers = []
        for i in range(num_conv_layers):
            conv_layer = nn.Sequential(nn.Conv2d(input_dim[-1] if i == 0 else conv_channels,
                                                 conv_channels, kernel_size, padding=kernel_size // 2),
                                       nn.BatchNorm2d(conv_channels),
                                       nn.ReLU(),
                                       nn.MaxPool2d(2))
            self.conv_layers.append(conv_layer)
        self.conv_layers = nn.Sequential(*self.conv_layers)

        self.fc1 = nn.Linear(conv_channels * wh_after_conv * wh_after_conv, embed_dim)

        self.fc2 = nn.Linear(embed_dim, 1)

    def forward_once(self, x):
        output = self.conv_layers(x)
        output = output.reshape(output.shape[0], -1)
        output = F.sigmoid(self.fc1(output))
        return output

    def forward(self, supportImage, supportLabel, queryImage, queryLabel):
        #query is the image the the one fixed
        query= self.forward_once(queryImage.permute(0, 3, 1, 2))

        support= self.forward_once(supportImage.permute(0, 3, 1, 2))

        support_repeat = support.repeat(queryImage.shape[0], 1)
        # 0, 1, 2, 0, 1, 2 ...
        query_rep = query.repeat_interleave(support.shape[0], dim=0)

        diff = torch.abs(query_rep - support_repeat)
        out = F.sigmoid(self.fc2(diff)).reshape(-1)

        label= (queryLabel.repeat_interleave(supportImage.shape[0]) == supportLabel.repeat(queryImage.shape[0])).float().reshape(-1)
        # print(label.shape)
        # print(query.shape)
        # print(support.shape)
        #print(queryLabel, supportLabel)
        loss = F.binary_cross_entropy_with_logits(out, label)

        # find best support sample for each query
        best_support = out.reshape(queryImage.shape[0], supportImage.shape[0]).argmax(-1)
        # predict class
        pred_y = supportLabel[best_support]

        return loss, pred_y

