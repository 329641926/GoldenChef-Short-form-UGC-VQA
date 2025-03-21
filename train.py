# -*- coding: utf-8 -*-
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random
from dataset import VideoDataset
from utils import performance_fit
from utils import plcc_loss, plcc_rank_loss, plcc_l1_loss
import UGCModel
from torchvision import transforms
import time
import pandas as pd


def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UGCModel.ViT_32_Swin_Tiny_Fast_densenet121_Model()

    pretrained_weights_path = config.pretrained_weights_path
    if pretrained_weights_path:
        model.load_state_dict(torch.load(pretrained_weights_path, map_location=device), strict=True)
        print("Successfully loaded pre-trained weights:", pretrained_weights_path)

    model = model.to(device)
    model = model.float()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.conv_base_lr, weight_decay=0.000001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)
    # optimizer = optim.Adam(model.parameters(), lr=config.conv_base_lr, weight_decay=0.000001)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=1e-6)

    if config.loss_type == 'plcc':
        criterion = plcc_loss
    elif config.loss_type == 'plcc_l1':
        criterion = plcc_l1_loss
    elif config.loss_type == 'plcc_rank':
        criterion = plcc_rank_loss
    elif config.loss_type == 'l2':
        criterion = nn.MSELoss().to(device)
    elif config.loss_type == 'l1':
        criterion = nn.L1Loss().to(device)
    elif config.loss_type == 'Huberloss':
        criterion = nn.HuberLoss().to(device)

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    transformations_train = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainset = VideoDataset(config.train_dir, config.train_datainfo, transformations_train, config.resize, config.frames)
    valset = VideoDataset(config.val_dir, config.val_datainfo, transformations_train, config.resize, config.frames)
    testset = VideoDataset(config.test_dir, config.test_datainfo, transformations_train, config.resize, config.frames)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=config.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=config.num_workers)

    best_val_criterion = -1  # SROCC min
    best_test_criterion = -1
    best_val = []
    best_test = []

    print('Starting training:')

    for epoch in range(config.epochs):
        model.train()
        batch_losses = []
        batch_losses_each_disp = []
        session_start_time = time.time()

        for i, (video, rs, mos, name) in enumerate(train_loader):
            video = video.to(device)
            rs = rs.to(device)
            labels = mos.to(device).float()
            outputs = model(video, rs)
            optimizer.zero_grad()
            loss_st = criterion(labels, outputs)
            loss = loss_st
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())
            loss.backward()
            optimizer.step()

            if (i + 1) % (config.print_samples // config.train_batch_size) == 0:
                session_end_time = time.time()
                avg_loss_epoch = sum(batch_losses_each_disp) / (config.print_samples // config.train_batch_size)

                print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' % (
                    epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size, avg_loss_epoch))

                batch_losses_each_disp = []

                print('CostTime: {:.4f}'.format(session_end_time - session_start_time))

                session_start_time = time.time()

        avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size)

        print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

        scheduler.step()

        lr = scheduler.get_last_lr()

        print('The current learning rate is {:.06f}'.format(lr[0]))
        # torch.save(model.state_dict(), config.Save_Path + '.pth')
        # print(f"模型权重已保存到 {config.Save_Path}_{epoch}")

        # do validation after each epoch
        with torch.no_grad():
            model.eval()

            label = np.zeros([len(valset)])
            y_output = np.zeros([len(valset)])

            for i, (video, rs, mos, name) in enumerate(val_loader):
                video = video.to(device)
                rs = rs.to(device)
                label[i] = mos.item()
                outputs = model(video, rs)
                y_output[i] = outputs.item()

            val_PLCC, val_SRCC, val_KRCC, val_RMSE = performance_fit(label, y_output)

            print(
                'Epoch {} completed. The result on the validation databaset: SRCC: {:.4f}, KRCC: {:.4f}, '
                'PLCC: {:.4f}, and RMSE: {:.4f}'.format(epoch + 1, val_SRCC, val_KRCC, val_PLCC, val_RMSE))

            if val_SRCC > best_val_criterion:
                print(" best_val_criterion in epoch {}".format(epoch + 1))

                best_val_criterion = val_SRCC
                best_val = [val_SRCC, val_KRCC, val_PLCC, val_RMSE]

                torch.save(model.state_dict(), config.Save_Path + '.pth')
                print(f"模型权重已保存到 {config.Save_Path}_val")

        if config.test:
            with torch.no_grad():
                model.eval()
                session_start_test = time.time()
                label = np.zeros([len(testset)])
                y_output = np.zeros([len(testset)])
                names = []

                for i, (video, rs, mos, name) in enumerate(test_loader):
                    video = video.to(device)
                    rs = rs.to(device)
                    label[i] = mos.item()
                    names.append('test/' + name[0] + '.mp4')
                    outputs = model(video, rs)

                    y_output[i] = outputs.item()
                    session_end_test = time.time()
                data = {
                    'filename': names,
                    'score': y_output
                }
                df = pd.DataFrame(data)
                df.to_csv(f'{config.Save_Path}_prediction_{epoch}.csv', index=False)
                print('CostTime: {:.4f}'.format(session_end_test - session_start_test))

                # test_PLCC, test_SRCC, test_KRCC, test_RMSE = performance_fit(label, y_output)

                # print( 'Epoch {} completed. Test_result: SRCC: {:.4f}, KRCC: {:.4f}, ' 'PLCC: {:.4f}, and RMSE:
                # {:.4f}'.format(epoch + 1, test_SRCC, test_KRCC, test_PLCC, test_RMSE))

                # if test_SRCC > best_test_criterion:
                #     print("best model best_test_criterion in epoch {}".format(epoch + 1))

                #     best_test_criterion = test_SRCC
                #     best_test = [test_SRCC, test_KRCC, test_PLCC, test_RMSE]
                #     # model_save_path = config.Save_Path
                #     # torch.save(model.state_dict(), model_save_path + '_best_test.pth')
                #     # print(f"模型权重已保存到 {model_save_path}_test")

    print('Training completed.')
    print(
        'The best training result on the base validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, '
        'and RMSE:'
        '{:.4f}'.format(best_val[0], best_val[1], best_val[2], best_val[3]))
    # print(
    #     'The best training result on the base validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, '
    #     'and RMSE:'
    #     '{:.4f}'.format(best_test[0], best_test[1], best_test[2], best_test[3]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str, default='train_video')
    parser.add_argument('--train_datainfo', type=str, default='data/train_data.csv')

    parser.add_argument('--val_dir', type=str, default='val_video')
    parser.add_argument('--val_datainfo', type=str, default='data/val_data.csv')

    parser.add_argument('--test_dir', type=str, default='test_video')
    parser.add_argument('--test_datainfo', type=str, default='data/test_data.csv')

    parser.add_argument('--print_samples', type=int, default=1000)
    parser.add_argument('--frames', type=int, default=12)  # frames

    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--test', action='store_true', default=True, help='Disable testing')

    # training parameters
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--train_batch_size', type=int, default=4)  # batch
    parser.add_argument('--conv_base_lr', type=float, default=2e-5)
    parser.add_argument('--decay_ratio', type=float, default=0.9)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--loss_type', type=str, default='plcc_rank')
    parser.add_argument('--Save_Path', type=str, default='Final_model_val')
    parser.add_argument('--pretrained_weights_path', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=8)

    config = parser.parse_args()
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)
    main(config)
