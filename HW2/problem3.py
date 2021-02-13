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
    flipped_points_idx = random.sample(range(0, 200), num_flips)
    for idx in flipped_points_idx:
        new_labels[idx] = 1 - new_labels[idx]   # Flips 0 to 1 and 1 to 0
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

    [m, b] = logistic_regression(random_points, labels_no_flips)
    print("Transition Layer (No flips): ", 1 / m.data[0][0].item())

    five_flipped_labels = flip_points(labels, 5)
    five_flipped_labels = torch.tensor(five_flipped_labels)
    [m, b] = logistic_regression(random_points, five_flipped_labels)
    print("Transition Layer (5 flips): ", 1 / m.data[0][0].item())

    fifteen_flipped_labels = flip_points(labels, 15)
    fifteen_flipped_labels = torch.tensor(fifteen_flipped_labels)
    [m, b] = logistic_regression(random_points, fifteen_flipped_labels)
    print("Transition Layer (15 flips): ", 1 / m.data[0][0].item())

    twenty_flipped_labels = flip_points(labels, 15)
    twenty_flipped_labels = torch.tensor(twenty_flipped_labels)
    [m, b] = logistic_regression(random_points, twenty_flipped_labels)
    print("Transition Layer (20 flips): ", 1 / m.data[0][0].item())

    twentyfive_flipped_labels = flip_points(labels, 15)
    twentyfive_flipped_labels = torch.tensor(twentyfive_flipped_labels)
    [m, b] = logistic_regression(random_points, twentyfive_flipped_labels)
    print("Transition Layer (25 flips): ", 1 / m.data[0][0].item())

    thirty_flipped_labels = flip_points(labels, 15)
    thirty_flipped_labels = torch.tensor(thirty_flipped_labels)
    [m, b] = logistic_regression(random_points, thirty_flipped_labels)
    print("Transition Layer (30 flips): ", 1 / m.data[0][0].item())

    thirtyfive_flipped_labels = flip_points(labels, 15)
    thirtyfive_flipped_labels = torch.tensor(thirtyfive_flipped_labels)
    [m, b] = logistic_regression(random_points, thirtyfive_flipped_labels)
    print("Transition Layer (35 flips): ", 1 / m.data[0][0].item())

if __name__ == "__main__":
    main()