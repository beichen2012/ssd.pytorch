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
from utils.torchutil import SaveCheckPoint
# from utils import make_dot
if not os.path.exists("./log"):
    os.mkdir("./log")
log = Logger("./log/{}_{}.log".format(__file__.split('/')[-1],
                                             time.strftime("%Y%m%d-%H%M%S"), time.localtime), level='debug').logger

pre_checkpoint = "./models/shufflenetv2_ssd_detach_20181201_iter_260000.pkl"
resume = False

USE_CUDA = True
GPU_ID = [0]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in GPU_ID])
device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")

train_batch = 32
display = 50

base_lr = 0.001
clip_grad = 20.0
momentum = 0.9
gamma = 0.1
weight_decay = 0.0005
stepsize = [150, 500, 700]
max_epoch = 800

save_interval = 10000

save_dir = "./models"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_prefix = save_dir + "/shufflenetv2_ssd_detach_20181207"
DATASET_ROOT = "~/dataset/VOCdevkit"
DATASET_ROOT = os.path.expanduser(DATASET_ROOT)

# data loader
def _worker_init_fn_():
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2**32-1
    random.seed(torch_seed)
    np.random.seed(np_seed)

def train():
    start_epoch = 0
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
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=momentum,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, stepsize, gamma)


    if USE_CUDA:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    if pre_checkpoint:
        cp = torch.load(pre_checkpoint)
        net.load_state_dict(cp['weights'])
        # net.load_state_dict(cp)
        log.info("=> load state dict from {}...".format(pre_checkpoint))
        if resume:
            optimizer.load_state_dict(cp['optimizer'])
            scheduler.load_state_dict(cp['scheduler'])
            start_epoch = cp['epoch']
            log.info("=> resume from epoch: {}, now the lr is: {}".format(start_epoch, optimizer.param_groups[0]['lr']))

    net.to(device)
    priorboxes = priorboxes.to(device)
    criterion.to(device)

    k = 0
    loc_loss = 0
    conf_loss = 0
    loss = 0
    for epoch in range(start_epoch, max_epoch + 1):
        # one eopch
        for batch_idx, (images, targets) in enumerate(data_loader):
            net.train()
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
            torch.nn.utils.clip_grad_norm_(net.parameters(), 20.0)
            optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            if k % display == 0:
                log.info("iter: {}, lr: {}, loss is: {:.4f}, loss_loc is: {:.4f}, loss_conf is: {:.4f}, time per iter: {:.4f} s".format(
                    k,
                    optimizer.param_groups[0]['lr'],
                    loss.item(),
                    loss_l.item(),
                    loss_c.item(),
                    t1-t0))
            if k % save_interval == 0:
                path = save_prefix + "_iter_{}.pkl".format(k)
                SaveCheckPoint(path, net, optimizer, scheduler, epoch)
                log.info("=> save model: {}".format(path))
            k += 1
        log.info('==> epoch: {}, lr: {}, loss is: {}'.format(epoch, optimizer.param_groups[0]['lr'], loss.item()))
        scheduler.step()

    log.info("optimize done...")
    path = save_prefix + "_final.pkl"
    SaveCheckPoint(path, net, optimizer, scheduler, max_epoch)
    log.info("=> save model: {} ...".format(path))


if __name__ == '__main__':
    train()
