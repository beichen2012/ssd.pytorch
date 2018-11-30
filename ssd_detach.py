# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import os

class SSDVGG16(nn.Module):
    def __init__(self, phase, size, num_classes):
        super(SSDVGG16, self).__init__()
        self.phase = phase
        self.size = size
        self.num_classes = num_classes

        # extra
        self.L2Norm = L2Norm(512, 20)

        # vgg base network
        self.vgg_conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU())
        self.vgg_conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU())
        self.vgg_conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU())
        self.vgg_conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU())
        self.vgg_conv5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU())
        self.fc6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=6, dilation=6)
        self.fc7 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        # extra
        self.conv6 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
                                   nn.ReLU())
        self.conv9 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
                                   nn.ReLU())

        # loc and conf
        mbox = [4, 6, 6, 6, 4, 4]
        self.loc = nn.ModuleList([nn.Conv2d(512, mbox[0] * 4, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(1024, mbox[1] * 4, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(512, mbox[2] * 4, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(256, mbox[3] * 4, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(256, mbox[4] * 4, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(256, mbox[5] * 4, kernel_size=3, stride=1, padding=1)])

        self.conf = nn.ModuleList([nn.Conv2d(512, mbox[0] * self.num_classes, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(1024, mbox[1] * self.num_classes, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(512, mbox[2] * self.num_classes, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(256, mbox[3] * self.num_classes, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(256, mbox[4] * self.num_classes, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(256, mbox[5] * self.num_classes, kernel_size=3, stride=1, padding=1)])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        sources = []
        loc_list = []
        conf_list = []
        # conv1
        x = self.vgg_conv1(x)
        x = self.max_pool1(x)

        # conv2
        x = self.vgg_conv2(x)
        x = self.max_pool1(x)

        # conv3
        x = self.vgg_conv3(x)
        x = self.max_pool2(x)

        # conv4
        x = self.vgg_conv4(x)
        sources += [self.L2Norm(x)]
        x = self.max_pool1(x)

        # conv5
        x = self.vgg_conv5(x)
        x = self.max_pool3(x)

        # fc6
        x = self.relu(self.fc6(x))
        # fc7
        x = self.relu(self.fc7(x))
        sources += [x]

        # extra
        ## conv6
        x = self.conv6(x)
        sources += [x]

        ## conv7
        x = self.conv7(x)
        sources += [x]

        ## conv8
        x = self.conv8(x)
        sources += [x]

        ## conv9
        x = self.conv9(x)
        sources += [x]

        # loc and conf
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc_list += [l(x).permute(0, 2, 3, 1).contiguous()]
            conf_list += [c(x).permute(0, 2, 3, 1).contiguous()]

        loc = torch.cat([o.view(o.size(0), -1) for o in loc_list], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf_list], 1)

        # output
        if self.phase == 'test':
            output = (loc.view(loc.size(0), -1, 4),
                      self.softmax(conf.view(conf.size(0), -1, self.num_classes))
                      )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes)
            )
        return output