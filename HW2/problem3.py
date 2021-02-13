import torch
import random
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

def flip_points(labels, num_flips):
    new_labels = labels
    positive_flips = 0
    negative_flips = 0
    for i in range(len(new_labels)):
        if positive_flips == num_flips and negative_flips == num_flips:
            break
        if positive_flips != num_flips and new_labels[i] == 1:
            new_labels[i] = 0
            positive_flips += 1
        elif negative_flips != num_flips and new_labels[i] == 0:
            new_labels[i] = 1
            negative_flips += 1
    return new_labels

def logistic_regression(points, labels):
    model = LogisticRegression(1, 2)
    cost_fn = torch.nn.CrossEntropyLoss()
    sgd_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(1000):
        sgd_optimizer.zero_grad()
        outputs = model(points)
        loss = cost_fn(outputs, labels)
        loss.backward()
        sgd_optimizer.step()
    return model.parameters()

def main():
    # Random points on a line centered at 0 with variance 1
    random_points = np.random.normal(0, 1, 200)
    labels = np.zeros(200)
    for i in range(len(random_points)):
        if random_points[i] > 0:
            labels[i] = 1

    random_points = random_points.astype(np.float32)
    labels = labels.astype(np.longlong)
    
    random_points = torch.tensor(random_points)
    random_points = random_points.reshape(-1, 1)
    labels_no_flips = torch.tensor(labels)

    transition_layer_widths = []
    num_flips = [0, 5, 15, 20, 25, 30, 35]
    for num_flip in num_flips:
        flipped_labels = flip_points(labels, num_flip)
        flipped_labels = torch.tensor(flipped_labels)
        [m, b] = logistic_regression(random_points, flipped_labels)
        transition_layer_widths.append(1 / m.data[0][0].item())

    plt.plot(num_flips, transition_layer_widths)
    plt.show()

if __name__ == "__main__":
    main()