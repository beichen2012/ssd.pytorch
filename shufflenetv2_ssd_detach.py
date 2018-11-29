# coding: utf-8
from collections import OrderedDict
import torch
import torch.nn as nn
from layers import *
from ShuffleNetV2 import _make_divisible, DownsampleUnit, BasicUnit

class ShuffleNetV2SSD_Detach(nn.Module):
    def __init__(self, phase, size, num_classes, scale=1):
        super(ShuffleNetV2SSD_Detach, self).__init__()
        self.phase = phase
        self.size = size
        self.num_classes = num_classes

        # shuffle net config
        self.scale = scale
        self.c_tag = 0.5
        self.groups = 2
        self.residual = False
        self.SE = False

        self.activation_type = nn.ReLU
        self.activation = nn.ReLU(inplace=True)

        self.num_of_channels = {0.5: [24, 48, 96, 192, 1024], 1: [24, 116, 232, 464, 1024],
                                1.5: [24, 176, 352, 704, 1024], 2: [24, 244, 488, 976, 2048]}
        self.c = [_make_divisible(chan, 2) for chan in self.num_of_channels[scale]]
        self.n = [3, 8, 3]  # TODO: should be [3,7,3]
        self.conv1 = nn.Conv2d(3, self.c[0], kernel_size=3, bias=False, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(self.c[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.shuffles = self._make_shuffles()

        self.conv_last = nn.Conv2d(self.c[-2], self.c[-1], kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm2d(self.c[-1])

        # L2 norm
        self.L2Norm = L2Norm(self.c[1], 20)

        # extra
        self.conv6 = nn.Sequential(nn.Conv2d(self.c[-1], 256, kernel_size=1, stride=1, padding=0),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
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
        self.loc = nn.ModuleList([nn.Conv2d(self.c[1], mbox[0] * 4, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(self.c[2], mbox[1] * 4, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(512, mbox[2] * 4, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(256, mbox[3] * 4, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(256, mbox[4] * 4, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(256, mbox[5] * 4, kernel_size=3, stride=1, padding=1)])

        self.conf = nn.ModuleList([nn.Conv2d(self.c[1], mbox[0] * self.num_classes, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(self.c[2], mbox[1] * self.num_classes, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(512, mbox[2] * self.num_classes, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(256, mbox[3] * self.num_classes, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(256, mbox[4] * self.num_classes, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(256, mbox[5] * self.num_classes, kernel_size=3, stride=1, padding=1)])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def _make_stage(self, inplanes, outplanes, n, stage):
        modules = OrderedDict()
        stage_name = "ShuffleUnit{}".format(stage)

        # First module is the only one utilizing stride
        first_module = DownsampleUnit(inplanes=inplanes, activation=self.activation_type, c_tag=self.c_tag,
                                      groups=self.groups)
        modules["DownsampleUnit"] = first_module
        second_module = BasicUnit(inplanes=inplanes * 2, outplanes=outplanes, activation=self.activation_type,
                                  c_tag=self.c_tag, SE=self.SE, residual=self.residual, groups=self.groups)
        modules[stage_name + "_{}".format(0)] = second_module
        # add more LinearBottleneck depending on number of repeats
        for i in range(n - 1):
            name = stage_name + "_{}".format(i + 1)
            module = BasicUnit(inplanes=outplanes, outplanes=outplanes, activation=self.activation_type,
                               c_tag=self.c_tag, SE=self.SE, residual=self.residual, groups=self.groups)
            modules[name] = module

        return nn.Sequential(modules)

    def _make_shuffles(self):
        modules = OrderedDict()
        stage_name = "ShuffleConvs"

        for i in range(len(self.c) - 2):
            name = stage_name + "_{}".format(i)
            module = self._make_stage(inplanes=self.c[i], outplanes=self.c[i + 1], n=self.n[i], stage=i)
            modules[name] = module

        return nn.Sequential(modules)

    def forward(self, x):
        sources = []
        loc_list = []
        conf_list = []
        # conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        # stage2
        x = self.shuffles[0](x)
        # priorbox
        sources += [self.L2Norm(x)]

        # stage3
        x = self.shuffles[1](x)
        # priorbox
        sources += [x]
        # stage4
        x = self.shuffles[2](x)
        # stage 5
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.activation(x)

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
