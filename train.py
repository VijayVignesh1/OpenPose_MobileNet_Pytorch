import argparse
import cv2
import os

import torch
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.coco import TrainDataset
from datasets.transformations import ConvertKeypoints
from modules.get_parameters import get_parameters_conv, get_parameters_bn, get_parameters_conv_depthwise
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.loss import l2_loss
from modules.load_state import load_state
from val import evaluate
import numpy as np
import config as args

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader


def train(prepared_train_labels, num_refinement_stages, base_lr, batch_size, batches_per_iter,
          num_workers, checkpoint_path, weights_only, checkpoints_folder, log_after,
          val_images_folder, val_output_name, checkpoint_after, val_after, val_file_name):

    if not os.path.exists(val_images_folder):
        os.makedirs(val_images_folder)
    net = PoseEstimationWithMobileNet(num_refinement_stages)
    stride = 8
    sigma = 7
    path_thickness = 1
    transform_data=transforms.Compose([ConvertKeypoints()])
    dataset = TrainDataset(prepared_train_labels,
                               stride, sigma, path_thickness,
                               transform=transform_data)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = optim.Adam([
        {'params': get_parameters_conv(net.model, 'weight')},
        {'params': get_parameters_conv_depthwise(net.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.model, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(net.cpm, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(net.cpm, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv_depthwise(net.cpm, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_conv(net.initial_stage, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(net.initial_stage, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(net.refinement_stages, 'weight'), 'lr': base_lr * 4},
        {'params': get_parameters_conv(net.refinement_stages, 'bias'), 'lr': base_lr * 8, 'weight_decay': 0},
        {'params': get_parameters_bn(net.refinement_stages, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.refinement_stages, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
    ], lr=base_lr, weight_decay=5e-4)

    num_iter = 0
    current_epoch = 0
    end_epoch=300
    drop_after_epoch = [100, 200, 220, 240, 260, 280]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.333)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        load_state(net, checkpoint)
        if not weights_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

            scheduler.load_state_dict(checkpoint['scheduler'])
            num_iter = checkpoint['iter']
            current_epoch = checkpoint['current_epoch']



    net = DataParallel(net).cuda()
    net.train()
    min_loss=np.inf
    print("Training Started...")
    for epochId in range(current_epoch, end_epoch):
        scheduler.step()
        total_losses = [0] * (num_refinement_stages + 1)  # heatmaps loss, paf loss per stage ## remove 1
        batch_per_iter_idx = 0
        for batch_data in train_loader:
            if batch_per_iter_idx == 0:
                optimizer.zero_grad()

            images = batch_data['image'].cuda()
            keypoint_maps = batch_data['keypoint_maps'].cuda()

            stages_output = net(images)

            losses = []

            for loss_idx in range(len(total_losses)):
                losses.append(l2_loss(stages_output[loss_idx], keypoint_maps, images.shape[0]))
                total_losses[loss_idx] += losses[-1].item() / batches_per_iter

            loss = losses[0]
            for loss_idx in range(1, len(losses)):
                loss += losses[loss_idx]
            loss /= batches_per_iter
            loss.backward()
            batch_per_iter_idx += 1
            if batch_per_iter_idx == batches_per_iter:
                optimizer.step()
                batch_per_iter_idx = 0
                num_iter += 1
            else:
                continue

            # evaluate(val_file_name, val_output_name, val_images_folder, net, 0)
            # net.train()
            # exit(0)
            if sum(total_losses)<min_loss:
                snapshot_name='{}/checkpoint_best.pth'.format(checkpoints_folder)
                torch.save({'state_dict': net.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'iter': num_iter,
                            'current_epoch': epochId},
                           snapshot_name) 
                min_loss=sum(total_losses)               
            if num_iter % log_after == 0:
                print('Iter: {}'.format(num_iter))
                for loss_idx in range(len(total_losses)):
                    print('\n'.join(['stage{}_heatmaps_loss: {}']).format(
                        loss_idx + 1, total_losses[loss_idx] / log_after))
                for loss_idx in range(len(total_losses)):
                    total_losses[loss_idx] = 0
            if num_iter % checkpoint_after == 0:
                snapshot_name = '{}/checkpoint_iter_{}.pth'.format(checkpoints_folder, num_iter)
                torch.save({'state_dict': net.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'iter': num_iter,
                            'current_epoch': epochId},
                           snapshot_name)
            if num_iter % val_after == 0:
                print('Validation...')
                evaluate(val_file_name, val_output_name, val_images_folder, net, num_iter)
                net.train()


if __name__ == '__main__':

    checkpoints_folder = '{}_checkpoints'.format(args.experiment_name)
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    train(args.prepared_train_labels, args.num_refinement_stages, args.base_lr, args.batch_size,
          args.batches_per_iter, args.num_workers, args.checkpoint_path, args.weights_only,
          checkpoints_folder, args.log_after, args.val_images_folder, args.val_output_name,
          args.checkpoint_after, args.val_after, args.val_file_name)
    print("Training Done...")
