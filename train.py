import torch
import numpy as np
from tqdm import tqdm

from device2use import device
from validate import validate


def train(data_loader, data_loader_test,
            resnet18, memnetwork,
             generator, discriminator,
             criterion, criterion_BCE, criterion_sL1,
             mem_opt, gen_opt, dis_opt,
             epochs, test_every=10,
             loss_threshold=.7):
    generator = generator.train()
    discriminator = discriminator.train()
    real_labels = torch.ones((2, 1)).to(device)
    fake_labels = torch.zeros((2, 1)).to(device)

    for epoch in range(epochs):
        print(f'Epoch {epoch}...')
        for data_instance in tqdm(data_loader):
            color_feature = data_instance['color_feat'].to(device)
            resnet_inp = data_instance['resnet_inp'].to(device)
            l_channel = (data_instance['l_channel'] / 100.).to(device)
            ab_channel = (data_instance['ab_channel'] / 110.).to(device)
            idx = data_instance['index'].to(device)

            resnet_out = resnet18(resnet_inp)
            loss = criterion(resnet_out, color_feature, loss_threshold)
            mem_opt.zero_grad()
            loss.backward()
            mem_opt.step() # only resnet

            with torch.no_grad():
                img_feature = resnet18(resnet_inp)
                memnetwork.update(img_feature, color_feature, loss_threshold, idx)

            # train disciminator
            dis_color_feat = torch.cat([torch.unsqueeze(color_feature, 2) for _ in range(256)], dim = 2)
            dis_color_feat = torch.cat([torch.unsqueeze(dis_color_feat, 3) for _ in range(256)], dim = 3)
            fake_ab_channel = generator(l_channel, color_feature)
            real = discriminator(ab_channel, l_channel, dis_color_feat)
            d_loss_real = criterion_BCE(real, real_labels)

            fake = discriminator(fake_ab_channel, l_channel, dis_color_feat)
            d_loss_fake = criterion_BCE(fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            gen_opt.zero_grad()
            dis_opt.zero_grad()
            d_loss.backward()
            dis_opt.step()

            # train generator
            fake_ab_channel = generator(l_channel, color_feature)
            fake = discriminator(fake_ab_channel, l_channel, dis_color_feat)

            g_loss_GAN = criterion_BCE(fake, real_labels)
            g_loss_smoothL1 = criterion_sL1(fake_ab_channel, ab_channel)
            g_loss = g_loss_GAN + g_loss_smoothL1

            dis_opt.zero_grad()
            gen_opt.zero_grad()
            g_loss.backward()
            gen_opt.step()

        if epoch % test_every == 0:
            generator.eval()
            validate(resnet18, memnetwork, generator, data_loader_test, epoch)
            generator.train()

    return l_channel, fake_ab_channel
