# coding: utf-8

from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from ssdshufflenetv2 import *
from utils import Logger, make_dot
log = Logger("{}.log".format(__file__.split('/')[-1], level='debug')).logger

USE_CUDA = True
device = torch.device('cuda' if USE_CUDA else 'cpu')
train_batch = 32
display = 10

base_lr = 0.01
momentum = 0.9
gamma = 0.1
weight_decay = 0.0005
stepsize = [5000, 30000, 60000, 80000, 100000]
max_iter = 120000

save_interval = 10000

save_dir = "./models"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_prefix = save_dir + "/shufflenetv2_ssd_20181126"
DATASET_ROOT = "/home/wyj/dataset/VOCdevkit"


if torch.cuda.is_available() and USE_CUDA:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def train():
    cfg = voc_shufflenetv2

    dataset = VOCDetection(root=DATASET_ROOT,
                           transform=SSDAugmentation(cfg['min_dim'],
                                                     MEANS))
    data_loader = data.DataLoader(dataset, train_batch,
                                  num_workers=1,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)


    ssd_net = build_shffulenetv2_ssd('train', cfg, 300, 21)
    net = ssd_net

    if USE_CUDA:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True
        net.cuda()
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=momentum,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, stepsize, gamma)


    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, USE_CUDA)

    num_epoch = max_iter // train_batch + 1
    k = 0
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    loss = 0
    for i in range(0, num_epoch):
        net.train()

        # one eopch
        for batch_idx, (images, targets) in enumerate(data_loader):
            images = images.to(device)
            targets = [i.to(device) for i in targets]

            # forward
            t0 = time.time()
            out = net(images)

            # back
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            scheduler.step()

            t1 = time.time()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            if k % display == 0:
                log.info("iter: {}, lr: {:.4f}, loss is: {:.4f}, loss_loc is: {:.4f}, loss_conf is: {:.4f}, time per iter: {:.4f} s".format(
                    k,
                    optimizer.param_groups[0]['lr'],
                    loss.item(),
                    loss_l.item(),
                    loss_c.item(),
                    t1-t0))
            if k % save_interval == 0:
                path = save_prefix + "_iter_{}.pkl".format(k)
                torch.save(net.to('cpu').state_dict(), path)
                net.to(device)
                log.info("save model: {}".format(path))
            k += 1
        log.info('epoch: {}, lr: {}, loss is: {}'.format(i, optimizer.param_groups[0]['lr'], loss.item()))
    log.info("optimize done...")
    path = save_prefix + "_final.pkl"
    torch.save(net.to('cpu').state_dict(), path)
    log.info("save model: {} ...".format(path))


if __name__ == '__main__':
    train()
