import torch
import torch.nn as nn


class RelationNetwork(torch.nn.Module):
    def __init__(self, input_dim, conv_channels, embed_dim,
                 kernel_size=5, num_conv_layers=2, num_pair_layers=2,
                 wh_after_conv=4):
        super().__init__()

        self.mse_loss = nn.MSELoss()

        self.conv_layers = []
        for i in range(num_conv_layers):
            conv_layer = nn.Sequential(nn.Conv2d(input_dim[-1] if i == 0 else conv_channels,
                                                 conv_channels, kernel_size, padding=kernel_size // 2),
                                       nn.BatchNorm2d(conv_channels),
                                       nn.ReLU(),
                                       nn.MaxPool2d(2))
            self.conv_layers.append(conv_layer)
        self.conv_layers = nn.Sequential(*self.conv_layers)

        self.embed_lin = nn.Linear(conv_channels * wh_after_conv * wh_after_conv, embed_dim)

        self.pair_lin_layers = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(),
                                             nn.Linear(embed_dim, 1), nn.Sigmoid())

    def forward(self, s_x, s_y, q_x, q_y):
        s_embeds = self.conv_layers(s_x.permute(0, 3, 1, 2))
        s_embeds = self.embed_lin(s_embeds.reshape(s_embeds.shape[0], -1))

        q_embeds = self.conv_layers(q_x.permute(0, 3, 1, 2))
        q_embeds = self.embed_lin(q_embeds.reshape(s_embeds.shape[0], -1))

        s_embeds_rep = s_embeds.repeat(q_x.shape[0], 1)
        # 0, 1, 2, 0, 1, 2 ...

        q_embeds_rep = q_embeds.repeat_interleave(s_x.shape[0], dim=0)
        # 0, 0, 0, 1, 1, 1, ...

        pair_embeds = torch.cat((q_embeds_rep, s_embeds_rep), dim=-1)

        pair_scores = self.pair_lin_layers(pair_embeds)

        pair_gold = (q_y.repeat_interleave(s_x.shape[0]) == s_y.repeat(q_x.shape[0])).float().reshape(-1)

        loss = self.mse_loss(pair_scores, pair_gold)

        # find best support sample for each query
        best_support = pair_scores.reshape(q_x.shape[0], s_x.shape[0]).argmax(-1)

        # predict class
        pred_y = s_y[best_support]

        return loss, pred_y
