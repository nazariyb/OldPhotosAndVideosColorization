import torch


def train(data_loader, resnet18, memnetwork, criterion, optimizer, loss_threshold=.7):
    for data_instance in data_loader:
        color_feature = data_instance['color_feat']
        resnet_inp = data_instance['resnet_inp']
        idx = data_instance['index']

        print('get resnet out')
        resnet_out = resnet18(resnet_inp)
        print('get loss')
        loss = criterion(resnet_out, color_feature, loss_threshold)
        print('zero grad')
        optimizer.zero_grad()
        # print('get resnet out')
        # loss.backward()
        print('optim step')
        optimizer.step() # only resnet

        with torch.no_grad():
            print('again resnet')
            img_feature = resnet18(resnet_inp)
            print('update')
            memnetwork.update(img_feature, color_feature, loss_threshold, idx)
        print('done')
        break