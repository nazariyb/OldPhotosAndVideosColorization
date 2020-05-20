import numpy as np
import torch
from tqdm import tqdm
from skimage.color import lab2rgb
from PIL import Image
import os

from device2use import device


def validate(resnet18, memory, generator, dataloader_test, epoch):
    with torch.no_grad():
        for n, data_instance in enumerate(tqdm(dataloader_test)):
            color_feature = data_instance['color_feat'].to(device)
            resnet_inp = data_instance['resnet_inp'].to(device)
            l_channel = (data_instance['l_channel'] / 100.).to(device)
            ab_channel = (data_instance['ab_channel'] / 110.).to(device)
            # idx = data_instance['index'].to(device)

            bs = resnet_inp.size()[0]

            resnet_out = resnet18(resnet_inp)
            top1_feature = memory.get_neighbors(resnet_out, 1)
            top1_feature = top1_feature[:, 0]
            result_ab_channel = generator(l_channel, memory.V[top1_feature])

            real_image = torch.cat((l_channel * 100, ab_channel * 100), dim=1).cpu().numpy()
            fake_image = torch.cat((l_channel * 100, result_ab_channel * 100), dim=1).cpu().numpy()
            gray_image = torch.cat((l_channel * 100, torch.zeros((bs, 2, 256, 256)).to(device)), dim=1).cpu().numpy()

            all_img = np.concatenate((real_image, fake_image, gray_image), axis=2)
            all_img = np.transpose(all_img, (0, 2, 3, 1))
            rgb_imgs = [lab2rgb(img) for img in all_img]
            rgb_imgs = np.array((rgb_imgs))
            rgb_imgs = (rgb_imgs * 255.).astype(np.uint8)

            for img in rgb_imgs:
                img = Image.fromarray(img)
                name = f'{epoch}-{n}-d_result.png'
                img.save(os.path.join('results', name))
