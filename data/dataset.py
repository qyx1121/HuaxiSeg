import cv2
import os
import os.path as osp
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import CenterCrop

import copy

def rotate_coordinate(x, y, angle, image_shape):
    theta_rad = np.deg2rad(-angle)
    
    center_x, center_y = (np.array(image_shape[::-1]) - 1) / 2.

    x_shifted = x - center_x
    y_shifted = y - center_y
    
    x_rotated = x_shifted * np.cos(theta_rad) - y_shifted * np.sin(theta_rad)
    y_rotated = x_shifted * np.sin(theta_rad) + y_shifted * np.cos(theta_rad)
    
    x_new = x_rotated + center_x
    y_new = y_rotated + center_y
    
    return x_new, y_new



def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, points, angle = 20):
    new_points = [0, 0, 0, 0]
    image_shape = (image.shape[0], image.shape[1])
    new_points[0], new_points[1] = rotate_coordinate(points[0], points[1], angle, image_shape)
    new_points[2], new_points[3] = rotate_coordinate(points[2], points[3], angle, image_shape)

    image = ndimage.rotate(image, angle, order=0, reshape=False)
    return image, new_points


def random_rotate_swinunet(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


# class RandomGenerator(object):
#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, image, label):

#         if random.random() > 0.5:
#             image, label = random_rot_flip(image, label)
#         elif random.random() > 0.5:
#             image, label = random_rotate(image, label)
#         x, y = image.shape
#         if x != self.output_size[0] or y != self.output_size[1]:
#             image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
#             label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
#         image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
#         label = torch.from_numpy(label.astype(np.float32))
        
#         sample = {'image': image, 'label': label.long()}
#         return sample

class RandomGeneratorEvans(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):

        label = copy.deepcopy(image)
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate_swinunet(image, label)
        return image

class RandomGeneratorACPC(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image, points):
        
        label = np.zeros((image.shape[0], image.shape[1]))
        label[round(points[1]), round(points[0])] = 1
        label[round(points[3]), round(points[2])] = 2

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
            p_1 = np.where(label == 1)
            p_2 = np.where(label == 2)
            points[0], points[1] = p_1[1][0], p_1[0][0]
            points[2], points[3] = p_2[1][0], p_2[0][0]
        elif random.random() > 0.5:
            angle = np.random.randint(-30, 30)
            image, points = random_rotate(image, points, angle)
        
        x, y, _ = image.shape        
        return image, points


class RandomGenerator(object):
    def __init__(self, output_size, num_classes = 3):
        self.output_size = output_size
        self.num_classes = num_classes

    def __call__(self, image, label):

        #if random.random() > 0.5:
           #image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate_swinunet(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))

        target = torch.zeros((self.num_classes,label.shape[0],label.shape[1]))
        label = label.to(torch.int64).unsqueeze(0)

        target.scatter_(dim = 0,index=label,value=1)

        sample = {'image': image, 'label': target.long()}
        return sample


class Huaxi_dataset(Dataset):
    def __init__(self, task, split, base_dir, img_dir, label_dir, transform = None):
        super().__init__()
        self.split = split
        self.transform = transform
        data_path = osp.join(base_dir, split + '.txt') if task not in ['bvr', 'ca'] else osp.join(base_dir, split + f'_{task}.txt')
        f = open(data_path).readlines()
        self.idx_list = [i.strip() for i in f]
        self.img_dir = img_dir
        self.label_dir = label_dir
    
    def __len__(self):
        return len(self.idx_list)
    
    def __getitem__(self, index):
        img_id = self.idx_list[index]
        img_path = osp.join(self.img_dir, img_id + ".jpg")
        if not osp.exists(img_path):
            img_path = img_path.replace("jpg", "png")

        #img_path = osp.join(self.img_dir, "image_%s.jpg"%str(img_id))
        
        img = cv2.imread(img_path,0)

        label_path = osp.join(self.label_dir, img_id + ".npy")
        #label_path = osp.join(self.label_dir, "image_%s_mask.npy"%str(img_id))
        label = np.load(label_path)

        if self.transform:
            sample = self.transform(img, label)

        return sample['image'], sample['label']