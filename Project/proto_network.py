import os
import time
import torch
import numpy as np
import multiprocessing as mp
from torch.autograd import Variable
from meta_learning import MetaLearning

dataset_directory_path = "omniglot/"

class ProtoNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = torch.nn.Sequential(
            self.convolution_block(input_dim[2], hidden_dim),
            self.convolution_block(hidden_dim, hidden_dim),
            self.convolution_block(hidden_dim, hidden_dim),
            self.convolution_block(hidden_dim, output_dim)
        )

    def loss(self, support_set, query_set, n_way, k_shot):
        x_support = [support_tuple[0] for support_tuple in support_set]
        y_support = [support_tuple[1] for support_tuple in support_set]
        x_query = [query_tuple[0] for query_tuple in query_set]
        y_query = [query_tuple[1] for query_tuple in query_set]

    def convolution_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

def main():
    train_directory_path = os.path.join(dataset_directory_path, "images_background")
    test_directory_path = os.path.join(dataset_directory_path, "images_evaluation")
    proto_net = ProtoNet((28, 28, 3), 64, 64)
    meta_learning = MetaLearning(proto_net, 10, 5)
    
    # Utilize multiprocessing since data loading takes >100s
    pool = mp.Pool(mp.cpu_count())
    start_time = time.time()
    results = pool.map(meta_learning.load_images, [train_directory_path, test_directory_path])
    end_time = time.time()
    pool.close()
    print("Elapsed time while loading data:", end_time - start_time)

    train_images_x, train_images_y, test_images_x, test_images_y = results[0][0], results[0][1], results[1][0], results[1][1]
    support_set, query_set = meta_learning.get_random_sample(train_images_x, train_images_y, 10, 5)
    
    proto_net.loss(support_set, query_set, 10, 5)

if __name__ == "__main__":
    main()