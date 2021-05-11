import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(input_dim[-1], 64, kernel_size=10, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1),

            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1),
            nn.ReLU(inplace=True),

        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_in')

        self.fc1 = nn.Linear(2304, 1200)

        self.fc2 = nn.Linear(1200, 1)

    def forward_once(self, x):
        output = self.cnn1(x)
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
        out = F.sigmoid(self.fc2(diff))

        label= (queryLabel.repeat_interleave(supportImage.shape[0]) == supportLabel.repeat(queryImage.shape[0])).float().reshape(-1)
        # print(label.shape)
        # print(query.shape)
        # print(support.shape)
        #print(queryLabel, supportLabel)
        loss = ContrastiveLoss()(query_rep, support_repeat, label)

        # find best support sample for each query
        best_support = out.reshape(queryImage.shape[0], supportImage.shape[0]).argmax(-1)
        # predict class
        pred_y = supportLabel[best_support]

        return loss, pred_y

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, query, support, label):
        euclidean_distance = F.pairwise_distance(query, support)
        loss_contrastive = torch.mean(label * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive