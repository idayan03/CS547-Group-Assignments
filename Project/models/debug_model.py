import torch
import torch.nn as nn

class DebugModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.cross_entropy = nn.CrossEntropyLoss()

        self.lin = nn.Linear(input_dim[0] * input_dim[1] * input_dim[2], num_classes)

    def forward(self, s_x, s_y, q_x, q_y):
        # who even needs the support set

        pred_scores = self.lin(q_x.reshape(q_x.shape[0], -1))

        pred_y = pred_scores.argmax(-1)

        loss = self.cross_entropy(pred_scores, q_y)

        return loss, pred_y
