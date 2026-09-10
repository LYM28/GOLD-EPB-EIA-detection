# GOLDtrain_unet.py（父类定义，必须确保包含以下代码）
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        # 关键修复：显式定义实例属性，供子类访问
        self.image_dir = image_dir  # 图像目录路径（必须定义）
        self.mask_dir = mask_dir  # 掩码目录路径（必须定义）

        # 获取文件名列表并排序（确保图像和掩码一一对应）
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.masks = [f for f in os.listdir(mask_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

        # 排序确保图像和掩码文件名对应
        self.images.sort()
        self.masks.sort()

        # 验证图像和掩码数量一致
        if len(self.images) != len(self.masks):
            raise ValueError(f"图像数量({len(self.images)})与掩码数量({len(self.masks)})不匹配！")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 使用实例属性拼接路径（子类将继承这些属性）
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # 读取图像和掩码
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 单通道掩码

        # 基础转换
        image = transforms.ToTensor()(image)
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask
