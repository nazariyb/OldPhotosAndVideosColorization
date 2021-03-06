import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from skimage import color
from torchvision import transforms as T

from utils import quantize_colors


class ColorDataset(torch.utils.data.Dataset):
    def __init__(self, root, needed_size, transforms=None):
        self.root = root
        self.transforms = transforms
        self.needed_size = needed_size
        self.imgs = list(sorted(os.listdir(os.path.join(root))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img_rgb = Image.open(img_path).convert("RGB")

        w, h = img_rgb.size
        if w != h:
            m = min(*img_rgb.size)
            img_rgb = T.transforms.CenterCrop(m)(img_rgb)
        img_rgb = img_rgb.resize((self.needed_size, self.needed_size), Image.LANCZOS)

        img_lab = color.rgb2lab(img_rgb)
        l = img_lab[:,:,:1]
        ab = img_lab[:,:,1:]
        color_feature = np.mean(quantize_colors(ab, 313), axis=(0,1))

        gray_image = [img_lab[:,:,:1]]
        h, w, c = img_lab.shape
        gray_image.append(np.zeros(shape = (h, w, 2)))
        gray_image = np.concatenate(gray_image, axis = 2)
        resnet_inp = color.lab2rgb(gray_image)
        resnet_inp = (resnet_inp - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        index = idx + 0.0

        img_item = {}
        img_item['l_channel'] = np.transpose(l, (2, 0, 1)).astype(np.float32)
        img_item['ab_channel'] = np.transpose(ab, (2, 0, 1)).astype(np.float32)
        img_item['color_feat'] = color_feature.astype(np.float32)
        img_item['resnet_inp'] = np.transpose(resnet_inp, (2, 0, 1)).astype(np.float32)
        img_item['index'] = np.array(([index])).astype(np.float32)[0]

        return img_item

    def __len__(self):
        return len(self.imgs)