import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class CelebA_loader(Dataset):
    def __init__(self, 
                 datapath='/root/mnt/datasets/facial_datasets/celebA/original', 
                 attribute_index = None,
                 transform=None):
        self.transform = transform
        self.attribute_index = attribute_index
        assert self.has_duplicates(self.attribute_index), 'Error: This attributes list has same index one or more.'
        
        ## load landmark information
        landmark_path = os.path.join(datapath, 'Anno/list_landmarks_align_celeba.txt')
        with open(landmark_path, 'r') as f:
            self.landmark_lines = f.readlines()[2:]
        
        ## load attribute index    
        attribute_path = os.path.join(datapath, 'Anno/list_attr_celeba.txt')
        with open(attribute_path, 'r') as f:
            self.attribute_lines = f.readlines()[2:]
        self.attributes = self.choose_attribute()
        
        self.image_path = os.path.join(datapath, 'img_celeba')
        self.image_items = os.listdir(self.image_path)
        #self.datalength = len(os.listdir(self.image_path))
        self.datalength = 100000
        
    def __len__(self):
        return self.datalength
    
    def has_duplicates(self, seq):
        return len(seq) == len(set(seq))
    
    def crop_image(self, index, image_array):
        img_item = self.landmark_lines[index].split()
        le_x, le_y = int(img_item[1]), int(img_item[2])
        re_x, re_y = int(img_item[3]), int(img_item[4])
        n_x, n_y = int(img_item[5]), int(img_item[6])
        lm_x, lm_y = int(img_item[7]), int(img_item[8])
        rm_x, rm_y = int(img_item[9]), int(img_item[10])
        median_x, median_y = (le_x+re_x+lm_x+rm_x)/4, (le_y+re_y+lm_y+rm_y)/4
        
        return image_array[int(median_y-64):int(median_y+64), int(median_x-64):int(median_x+64)]
    
    def choose_attribute(self):
        attrs = []
        for item in self.attribute_lines:
            item_list = item.split()[1:]
            attrs.append([int(item_list[i]) if int(item_list[i])==1 else 0 for i in self.attribute_index])
        return attrs
    
    def __getitem__(self, i):
        img = Image.open(os.path.join(self.image_path, self.image_items[i]))
        img_array = np.array(img)
        croped_img = self.crop_image(i, img_array)
        img_tensor = torch.from_numpy(croped_img.transpose(2, 0, 1)) / 255.
        attr = torch.tensor(self.attributes[i]).float()
        
        return img_tensor, attr