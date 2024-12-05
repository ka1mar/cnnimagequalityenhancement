import warnings
warnings.filterwarnings('ignore')


import numpy as np
import os

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class PReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(PReLUBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x

class EnhancerModel(nn.Module):
    def __init__(self):
        super(EnhancerModel, self).__init__()
        self.conv_1 = PReLUBlock(3, 128, (9, 9), padding=4)
        self.conv_2 = PReLUBlock(128, 64, (7, 7), padding=3)
        self.conv_3 = PReLUBlock(64, 64, (3, 3), padding=1)
        self.conv_4 = PReLUBlock(64, 64, (3, 3), padding=1)
        self.conv_5 = PReLUBlock(64, 32, (1, 1), padding=0)

        self.conv_11 = PReLUBlock(3, 128, (9, 9), padding=4)
        self.conv_22 = PReLUBlock(256, 64, (7, 7), padding=3)
        self.conv_33 = PReLUBlock(128, 64, (3, 3), padding=1)
        self.conv_44 = PReLUBlock(128, 64, (3, 3), padding=1)
        self.conv_55 = PReLUBlock(128, 32, (1, 1), padding=0)

        self.conv_out = nn.Conv2d(64, 3, (5, 5), padding=2)

    def forward(self, x):
        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(conv_1)
        conv_3 = self.conv_3(conv_2)
        conv_4 = self.conv_4(conv_3)
        conv_5 = self.conv_5(conv_4)

        conv_11 = self.conv_11(x)
        feat_11 = torch.cat([conv_1, conv_11], dim=1)
        conv_22 = self.conv_22(feat_11)
        feat_22 = torch.cat([conv_2, conv_22], dim=1)
        conv_33 = self.conv_33(feat_22)
        feat_33 = torch.cat([conv_3, conv_33], dim=1)
        conv_44 = self.conv_44(feat_33)
        feat_44 = torch.cat([conv_4, conv_44], dim=1)
        conv_55 = self.conv_55(feat_44)
        feat_55 = torch.cat([conv_5, conv_55], dim=1)

        conv_10 = self.conv_out(feat_55)
        output_tensor = x + conv_10
        return output_tensor

class ImageDataset(Dataset):
    def __init__(self, folder_raw, folder_comp, transform=None, augment=None):
        self.folder_raw = folder_raw
        self.folder_comp = folder_comp
        self.transform = transform
        self.augment = augment
        self.raw_images = [f for f in os.listdir(folder_raw) if os.path.isfile(os.path.join(folder_raw, f))]

    def __len__(self):
        return len(self.raw_images)

    def __getitem__(self, idx):
        raw_image_path = os.path.join(self.folder_raw, self.raw_images[idx])
        comp_image_path = os.path.join(self.folder_comp, self.raw_images[idx])

        raw_image = np.asarray(Image.open(raw_image_path))
        comp_image = np.asarray(Image.open(comp_image_path))

        if self.augment:
            comp_image = self.augment(image=comp_image)["image"]

        if self.transform:
            transformed_images = self.transform(image=raw_image, comp_image=comp_image)
            raw_image, comp_image = transformed_images["image"] / 255., transformed_images["comp_image"] / 255.
    
        return comp_image, raw_image

def psnr(prediction, target):
    mse = nn.functional.mse_loss(prediction, target)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
