import os
import cv2
import numpy as np
from scipy import ndimage

class MetaLearning:
    """Meta Learning Framework for the 4 models
    """
    def __init__(self, model, n_way, k_shot):
        self.model = model
        self.n_way = n_way
        self.k_shot = k_shot

    def train():
        return
    
    def evaluate():
        return

    def load_images(self, directory_path):
        """Returns the characters of the Omniglot dataset as a numpy array
        (Note: The character classes have been augmented with rotations in multiples of 90 degrees 
        similar to what the Prototypical Network paper did)

        Args:
            directory_path: the path to the train or test directory

        Returns:
            images_x: a numpy array of the features of the images
            images_y: a numpy array of the labels of the images
        """
        images_x = []
        images_y = []
        for alphabet in os.listdir(directory_path):
            if alphabet.startswith('.'):    # Directory contains .DS_Store hidden file so ignore
                continue
            alphabet_path = os.path.join(directory_path, alphabet)
            for character in os.listdir(alphabet_path):
                if character.startswith('.'):    # Directory contains .DS_Store hidden file so ignore
                    continue
                character_path = os.path.join(alphabet_path, character)
                for image_name in os.listdir(character_path):
                    image_path = os.path.join(character_path, image_name)
                    image = cv2.resize(cv2.imread(image_path), (28, 28))    # The Proto Paper resizes this to (28, 28) might not apply to the other models
                    rotated_90 = ndimage.rotate(image, 90)
                    rotated_180 = ndimage.rotate(image, 180)
                    rotated_270 = ndimage.rotate(image, 270)
                    images_x.extend((image, rotated_90, rotated_270, rotated_180))
                    images_y.extend((
                        alphabet + '_' + character + '_0',
                        alphabet + '_' + character + '_90',
                        alphabet + '_' + character + '_180',
                        alphabet + '_' + character + '_270'
                    ))
        return np.array(images_x), np.array(images_y)

    def get_random_sample(self, images_x, images_y, n_way, k_shot):
        """Returns a random sample of size n_way * k_shot for N-way, K-shot episode based learning

        Args:
            images_x:
            images_y:
            n_way:
            k_shot:

        Return:
            support_set:
            query_set:
        """
        support_set = []
        query_set = []
        characters = np.random.choice(np.unique(images_y), n_way, replace=False)
        for character in characters:
            images_of_character = images_x[images_y == character]
            permuted_images_of_character = np.random.permutation(images_of_character)
            support = permuted_images_of_character[:k_shot]
            query = permuted_images_of_character[k_shot:(2 * k_shot)]
            support = [(x, character) for x in support]
            query = [(x, character) for x in query]
            support_set.extend(support)
            query_set.extend(query)
        return support_set, query_set
