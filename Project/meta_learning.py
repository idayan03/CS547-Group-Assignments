import os
import cv2
import numpy as np
from scipy import ndimage

import hydra

import time
import multiprocessing as mp

import torch
import torch.nn.functional as F


class MetaLearning:
    """Meta Learning Framework for the 4 models
    """

    def __init__(self, cfg):

        self.cfg = cfg

        self.model = hydra.utils.instantiate(config=cfg.model)

        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot

        train_path = cfg.data.train_path
        test_path = cfg.data.test_path

        # Utilize multiprocessing since data loading takes >100s
        # print("CPU count:", mp.cpu_count())
        # pool = mp.Pool(mp.cpu_count())
        start_time = time.time()

        self.train_images_x, self.train_images_y = self.load_images(train_path, True)
        self.test_images_x, self.test_images_y = self.load_images(test_path, False)

        # (self.train_images_x, self.train_images_y), \
        # (self.test_images_x, self.test_images_y) = \
        #    pool.starmap(self.load_images, [(train_path, True), (test_path, False)])

        end_time = time.time()
        # pool.close()

        print("Elapsed time while loading data:", end_time - start_time)

        if cfg.use_gpu:
            self.model = self.model.cuda()

    def train(self):

        trainer_cfg = self.cfg.trainer

        optimizer = hydra.utils.instantiate(trainer_cfg.optimizer, self.model.parameters())

        for train_ep_id in range(1, trainer_cfg.train_episodes + 1):

            self.model.train()

            train_s_x, train_s_y, train_q_x, train_q_y = self.get_random_sample(self.train_images_x,
                                                                                self.train_images_y)

            loss, _ = self.run_episode(train_s_x, train_s_y, train_q_x, train_q_y)

            if train_ep_id % 100 == 0:
                print(train_ep_id, "loss:", loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if train_ep_id % trainer_cfg.eval_every == 0:

                print("Evaluating...")

                self.model.eval()

                total_eval_acc = 0

                for eval_ep_id in range(1, trainer_cfg.eval_episodes):
                    test_s_x, test_s_y, test_q_x, test_q_y = self.get_random_sample(self.test_images_x,
                                                                                    self.test_images_y)

                    _, test_pred = self.run_episode(test_s_x, test_s_y, test_q_x, test_q_y)

                    total_eval_acc += (test_pred == test_q_y).sum() / float(test_q_y.shape[0])

                total_eval_acc /= trainer_cfg.eval_episodes

                print("Eval accuracy after", train_ep_id, "training episodes:", total_eval_acc)

                if trainer_cfg.lr_decay:
                    optimizer.param_groups[0]['lr'] *= 0.5

        return

    def run_episode(self, s_x, s_y, q_x, q_y):
        #want to get loss, pred_y
        return self.model(s_x, s_y, q_x, q_y)

    def load_images(self, data_path, is_train=False):
        """Returns the characters of the Omniglot dataset as a numpy array
        (Note: The character classes have been augmented with rotations in multiples of 90 degrees
        similar to what the Prototypical Network paper did)


        Returns:
            images_x: a numpy array of the features of the images
            images_y: a numpy array of the labels of the images
        """
        images_x = []
        images_y = []
        for alphabet in os.listdir(data_path):
            if alphabet.startswith('.'):  # Directory contains .DS_Store hidden file so ignore
                continue
            alphabet_path = os.path.join(data_path, alphabet)
            for character in os.listdir(alphabet_path):
                if character.startswith('.'):  # Directory contains .DS_Store hidden file so ignore
                    continue
                character_path = os.path.join(alphabet_path, character)
                for image_name in os.listdir(character_path):
                    image_path = os.path.join(character_path, image_name)
                    image = cv2.resize(cv2.imread(image_path), (28, 28))
                    # The Proto Paper resizes this to (28, 28) might not apply to the other models
                    images_x.append(image)
                    images_y.append(alphabet + '_' + character)

                    if is_train and self.cfg.data.augment_rotate:
                        rotated_90 = ndimage.rotate(image, 90)
                        rotated_180 = ndimage.rotate(image, 180)
                        rotated_270 = ndimage.rotate(image, 270)
                        images_x.extend((rotated_90, rotated_270, rotated_180))
                        images_y.extend((
                            alphabet + '_' + character + '_90',
                            alphabet + '_' + character + '_180',
                            alphabet + '_' + character + '_270'
                        ))

        return np.array(images_x), np.array(images_y)

    def get_random_sample(self, x, y):
        """Returns a random sample of size n_way * k_shot for N-way, K-shot episode based learning

        Return:
            support_set:
            query_set:
        """
        support_set_x, support_set_y, query_set_x, query_set_y = [], [], [], []
        characters = np.random.choice(np.unique(y), self.n_way, replace=False)
        char_to_id = dict()

        for character in characters:
            char_to_id[character] = len(char_to_id)
            images_of_character = x[y == character]
            permuted_images_of_character = np.random.permutation(images_of_character)
            support = permuted_images_of_character[:self.k_shot]
            query = permuted_images_of_character[self.k_shot:(2 * self.k_shot)]

            support_set_x.extend(support)
            query_set_x.extend(query)

            support_set_y.extend([char_to_id[character] for x in support])
            query_set_y.extend([char_to_id[character] for x in query])

        support_set_x = torch.Tensor(support_set_x)
        query_set_x = torch.Tensor(query_set_x)

        support_set_y = torch.LongTensor(support_set_y)
        query_set_y = torch.LongTensor(query_set_y)

        if self.cfg.use_gpu:
            support_set_x = support_set_x.cuda()
            support_set_y = support_set_y.cuda()

            query_set_x = query_set_x.cuda()
            query_set_y = query_set_y.cuda()

        return support_set_x, support_set_y, query_set_x, query_set_y
