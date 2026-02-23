import random

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ColorJitter, functional, Compose
import torch.nn.functional as F

class ImageAugmentor:
    def __init__(self, saturation_range=[0.75,1.25], gamma=[1,1,1,1]):
        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness=0, contrast=0, saturation=saturation_range, hue=0.05/3.14)])
        self.eraser_aug_prob = 0.5

    def color_transform(self, img):
        """ Photometric augmentation """
        img = np.array(self.photo_aug(Image.fromarray(img)), dtype=np.uint8)

        return img

    def eraser_transform(self, img, bounds=[50, 100]):
        """ Occlusion augmentation """
        ht, wd = img.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            max_brightness = np.max(img.reshape(-1, 3), axis=0)   # To emulate glare
            x0 = np.random.randint(0, wd)
            y0 = np.random.randint(0, ht)
            dx = np.random.randint(bounds[0], bounds[1])
            dy = np.random.randint(bounds[0], bounds[1])
            img[y0:y0+dy, x0:x0+dx, :] = max_brightness

        return img

    def __call__(self, img):
        img = self.color_transform(img)
        img = self.eraser_transform(img)

        return img