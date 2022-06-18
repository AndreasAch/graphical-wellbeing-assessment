import os
from os import listdir

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DoodleDatasetSimple(Dataset):
    """
    Class that prepares the dataset for loading
    :param doodle_path: The path where the images are located
    :param transform: The transformation to be applied to each image
    :param translation: The dictionary to match each of the images to its label
    :return An image ready to be fed into the model and its corresponding class label
    """
    def __init__(self, doodle_path, transform, translation):
        self.path = doodle_path
        self.folder = [x for x in listdir(doodle_path)]
        self.transform = transform
        self.translation_dict = translation

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.path, self.folder[idx])
        image = Image.open(img_loc).convert('RGB')
        single_img = self.transform(image)

        imageClass = self.translation_dict[self.folder[idx]]
        sample = {'image': torch.from_numpy(np.array(single_img)),
                  'class': imageClass}
        return sample

