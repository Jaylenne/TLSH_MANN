from curses import meta
import os
import random
import numpy as np
from torchvision import transforms
from utils import *
from more_itertools import unzip

class DataGenerator(object):

    def __init__(self, ways, shots, config={}):

        self.ways = ways
        self.shots = shots
        
        data_folder = config.get('data_folder', './omniglot/python/dataset')

        self.image_size = config.get('image_size', (32, 32))
        self.input_dim = np.prod(self.image_size)
        self.output_dim = self.ways

        character_folder = [os.path.join(data_folder, alphabet, character)
                            for alphabet in os.listdir(data_folder)
                            for character in os.listdir(os.path.join(data_folder, alphabet))]
        
        random.seed(1)
        random.shuffle(character_folder)

        meta_train = 964 # training characters = 964
        meta_val = 250

        self.meta_train_characters = character_folder[: meta_train]
        self.meta_val_characters = character_folder[meta_train: meta_train + meta_val]
        self.test_characters = character_folder[meta_train + meta_val:]

        self.train_data_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                        transforms.RandomAffine(30, (0.15, 0.15))])
                                                        
        self.val_data_transform = transforms.Compose([transforms.Resize([32, 32])])

    def sample_batch(self, batch_type, batch_size, _rand=True):
        if batch_type == "train":
            folders = self.meta_train_characters
            transform = self.train_data_transform
        elif batch_type == 'val':
            folders = self.meta_val_characters
            transform = self.val_data_transform
        elif batch_type == 'test':
            folders = self.test_characters
            transform = self.val_data_transform
        if _rand:
          sample_classes = random.sample(folders, self.ways)
        else:
          sample_classes = folders[:self.ways]
        one_hot_labels = np.identity(self.ways)

        query_candidates = []
        support_images = []
        support_labels = []
        for i, path in zip(one_hot_labels, sample_classes):
          all_images = os.listdir(path)
          support_idx = random.sample(range(len(all_images)), self.shots)
          for idx in range(len(all_images)):
            if idx in support_idx:
              support_images.append(image_file_to_array(os.path.join(path, all_images[idx]), transform))
              support_labels.append(i)
            else:
              query_candidates.append((i,os.path.join(path, all_images[idx])))

        support_labels = np.vstack(support_labels)
        support_images = np.array(support_images)

        query_labels, query_images = unzip(random.sample(query_candidates, batch_size))
        query_labels = np.vstack(query_labels)
        query_images = np.array([image_file_to_array(i, transform) for i in query_images])

        return support_labels, support_images, query_labels, query_images
