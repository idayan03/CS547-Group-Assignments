import torch
import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        outputs = self.linear(x)
        outputs = self.sigmoid(outputs)
        return outputs

def flip_points(labels, num_flips):
    """ Flips num_flips points on each side of the origin to the wrong label.

    Args:
        labels: The labels of each point.
        num_flips: The number of flips on each side of the origin.

    Returns:
        The new flipped labels.
    """
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
    """ Performs logistic regression similar to professor's model.

    Args:
        points: The randomly selected gaussian points.
        labels: The labels of the gaussian points (1 if to the right of origin and 0 if to the left of origin).

    Returns:
        The trained logistic model
    """
    model = LogisticRegression(1, 1)
    cost_fn = torch.nn.BCELoss()
    sgd_optimizer = torch.optim.SGD(model.parameters(), lr=0.075)
    for epoch in range(1000):
        sgd_optimizer.zero_grad()
        outputs = model(points)
        loss = cost_fn(outputs, labels)
        loss.backward()
        sgd_optimizer.step()
    return model

def main():
    # Random points on a line centered at 0 with variance 1
    random_points = np.random.normal(0, 1, 200)
    labels = np.zeros(200)
    for i in range(len(random_points)):
        if random_points[i] > 0:
            labels[i] = 1

    random_points = random_points.astype(np.float32)
    labels = labels.astype(np.float32)
    
    random_points = torch.tensor(random_points)
    random_points = random_points.reshape(-1, 1)

    # Performing Logistic Regression with sklearn (Only done to verify our model is correct)
    lr = sklearn.linear_model.LogisticRegression(solver="lbfgs", random_state=0).fit(random_points, labels)
    (m_sk,b_sk) = (lr.coef_.item(),lr.intercept_.item())
    print("Results of sklearn.linear_model.LogisticRegression: $m_{{sp}}={0:.2f}$, $b_{{sp}}={1:.2f}$".format(m_sk,b_sk))

    transition_layer_widths = []
    num_flips = [0, 5, 15, 20, 25, 30, 35]
    for num_flip in num_flips:
        flipped_labels = flip_points(labels, num_flip)
        flipped_labels = torch.tensor(flipped_labels).reshape(-1, 1)
        model = logistic_regression(random_points, flipped_labels)
        (m_pt, b_pt) = (model.linear.weight.item(), model.linear.bias.item())
        if num_flip == 0:
            print("Results of pytorch: $m_{{pt}}={0:.2f}$, $b_{{pt}}={1:.2f}$".format(m_pt, b_pt))
        transition_layer_widths.append(1 / m_pt)

    plt.plot(num_flips, transition_layer_widths)
    plt.xlabel("No. of flips")
    plt.ylabel("Transition Layer Width")
    plt.show()

if __name__ == "__main__":
    main()