# GOLDtrain_unet.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        
        self.image_dir = image_dir 
        self.mask_dir = mask_dir  

        
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.masks = [f for f in os.listdir(mask_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

        
        self.images.sort()
        self.masks.sort()

        
        if len(self.images) != len(self.masks):
            raise ValueError(f"图像数量({len(self.images)})与掩码数量({len(self.masks)})不匹配！")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  

        
        image = transforms.ToTensor()(image)
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask
