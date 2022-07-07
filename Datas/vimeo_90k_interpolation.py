import os
import random

import numpy as np
import torch
from scipy.ndimage import imread

def image_preprocessing(image_path, h_offset, w_offset, frame_size, use_flip):
    image = imread(image_path)
    image = image[h_offset:h_offset + frame_size[0], w_offset: w_offset + frame_size[1], :]
    if use_flip[0]:
        image = np.fliplr(image)
    if use_flip[1]:
        image = np.flipud(image)
    image = np.transpose(image, (2, 0, 1)).astype("float32")/ 255.0
    return image

def Vimeo_90K_loader(root_path, image_path, frame_size = (256, 448), data_aug = True):
    triplet_images_folder = os.path.join(root_path, image_path)
    image_name_list = ["im1.png", "im2.png", "im3.png"]
    use_flip = [False, False] # lr flip, ud flip
    if data_aug and random.randint(0, 1):
        image_name_list = image_name_list[::-1]
        use_flip = [random.randint(0, 1), random.randint(0, 1)]
        
    images = list()  
    h_offset = random.choice(range(256 - frame_size[0] + 1))
    w_offset = random.choice(range(448 - frame_size[1] + 1))
    for image_name in image_name_list:
        image_path = os.path.join(triplet_images_folder, image_name)
        image = image_preprocessing(image_path, h_offset, w_offset, frame_size, use_flip)
        images.append(image)
    return images

class Vimeo90KDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, file_list, loader = Vimeo_90K_loader):
        self.root_path = root_path
        self.file_list = file_list
        self.loader = loader

    def __getitem__(self, index):
        image_t0, image_t2, image_gt = self.loader(self.root_path, self.file_list[index])
        return image_t0, image_t2, image_gt

    def __len__(self):
        return len(self.file_list)

def make_dataset(file_list, use_shuffle = True):
    if os.path.exists(file_list):
        img_list = open(os.path.join(file_list)).read().splitlines()    
    else:
        print(file_list + ' not exist!')
        raise NotImplementedError
    if use_shuffle:
        random.shuffle(img_list)
    return img_list

def Vimeo_90K_dataset_generation(root_path, train_list_txt = None, test_list_txt = None):
    if train_list_txt is not None:
        train_list = make_dataset(train_list_txt, use_shuffle = True)
        train_dataset = Vimeo90KDataset(root_path, train_list,  loader = Vimeo_90K_loader)
    else:
        print('train list lost!')
        raise NotImplementedError
        
    if test_list_txt is not None:
        test_list = make_dataset(test_list_txt, use_shuffle = False)
        test_dataset = Vimeo90KDataset(root_path, test_list,  loader = Vimeo_90K_loader)
    else:
        print('test list lost!')
        raise NotImplementedError
    
    return train_dataset, test_dataset