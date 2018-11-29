# coding: utf-8

from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
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
import random
from shufflenetv2_ssd_detach import *
from utils import Logger
# from utils import make_dot
log = Logger("{}.log".format(__file__.split('/')[-1], level='debug')).logger

USE_CUDA = True
GPU_ID = [0]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in GPU_ID])
device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")

train_batch = 32
display = 10

base_lr = 0.01
momentum = 0.9
gamma = 0.1
weight_decay = 0.0005
stepsize = [5000, 50000, 100000, 120000, 140000]
max_iter = 150000

save_interval = 10000

save_dir = "./models"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_prefix = save_dir + "/shufflenetv2_ssd_detach_20181126"
DATASET_ROOT = "/home/beichen2012/dataset/VOCdevkit"

# data loader
def _worker_init_fn_():
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2**32-1
    random.seed(torch_seed)
    np.random.seed(np_seed)
def train():
    cfg = voc_shufflenetv2

    dataset = VOCDetection(root=DATASET_ROOT,
                           transform=SSDAugmentation(cfg['min_dim'],
                                                     MEANS))
    data_loader = data.DataLoader(dataset, train_batch,
                                  num_workers=1,
                                  shuffle=True, collate_fn=detection_collate,
                                  worker_init_fn=_worker_init_fn_(),
                                  pin_memory=True)

    # net
    ssd_net = ShuffleNetV2SSD_Detach("train", cfg["min_dim"], cfg["num_classes"], 1.5)
    net = ssd_net

    # priorbox
    net_priorbox = PriorBox(cfg)
    with torch.no_grad():
        priorboxes = net_priorbox.forward()

    # criterion
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, device)


    if USE_CUDA:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True
    net.to(device)
    priorboxes = priorboxes.to(device)
    criterion.to(device)

    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=momentum,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, stepsize, gamma)


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
            loc, conf = out
            loss_l, loss_c = criterion((loc, conf, priorboxes), targets)
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
