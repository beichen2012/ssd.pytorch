# coding: utf-8
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from layers import *
from ShuffleNetV2 import _make_divisible, DownsampleUnit, BasicUnit


class LocAndConf(nn.Module):
    def __init__(self, c_in, c_out, num_classes):
        super(LocAndConf, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.num_classes = num_classes

        self.conv_loc = nn.Conv2d(c_in, c_out * 4, kernel_size=3, padding=1)
        self.conv_conf = nn.Conv2d(c_in, c_out * num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        loc = self.conv_loc(x)
        conf = self.conv_conf(x)
        return loc, conf


class ShuffleNetV2SSD(nn.Module):
    def __init__(self, phase, size, num_classes, cfg, scale=1):
        super(ShuffleNetV2SSD, self).__init__()
        self.phase = phase
        self.size = size
        self.num_classes = num_classes
        self.scale = scale
        self.cfg = cfg
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

        # priorbox
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = self.priorbox.forward()

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

        # extra
        self.conv6_1 = nn.Conv2d(self.c[-1], 256, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv8_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv8_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.conv9_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)

        # loc and conf
        mbox = [4, 6, 6, 6, 4, 4]
        self.lc_1 = LocAndConf(self.c[1], mbox[0], self.num_classes)
        self.lc_2 = LocAndConf(self.c[2], mbox[1], self.num_classes)
        self.lc_3 = LocAndConf(512, mbox[2], self.num_classes)
        self.lc_4 = LocAndConf(256, mbox[3], self.num_classes)
        self.lc_5 = LocAndConf(256, mbox[4], self.num_classes)
        self.lc_6 = LocAndConf(256, mbox[5], self.num_classes)


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
        # conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        # stage2
        x = self.shuffles[0](x)
        # priorbox
        b1 = self.L2Norm(x)

        # stage3
        x = self.shuffles[1](x)
        # priorbox
        b2 = x
        # stage4
        x = self.shuffles[2](x)
        # stage 5
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.activation(x)

        # extra
        x = self.activation(self.conv6_1(x))
        x = self.activation(self.conv6_2(x))
        b3 = x

        x = self.activation(self.conv7_1(x))
        x = self.activation(self.conv7_2(x))
        b4 = x

        x = self.activation(self.conv8_1(x))
        x = self.activation(self.conv8_2(x))
        b5 = x

        x = self.activation(self.conv9_1(x))
        x = self.activation(self.conv9_2(x))
        b6 = x

        # loc and conf
        loc1, conf1 = self.lc_1(b1)
        loc2, conf2 = self.lc_2(b2)
        loc3, conf3 = self.lc_3(b3)
        loc4, conf4 = self.lc_4(b4)
        loc5, conf5 = self.lc_5(b5)
        loc6, conf6 = self.lc_6(b6)

        ## permute
        loc1 = loc1.permute(0,2,3,1).contiguous()
        loc2 = loc2.permute(0, 2, 3, 1).contiguous()
        loc3 = loc3.permute(0, 2, 3, 1).contiguous()
        loc4 = loc4.permute(0, 2, 3, 1).contiguous()
        loc5 = loc5.permute(0, 2, 3, 1).contiguous()
        loc6 = loc6.permute(0, 2, 3, 1).contiguous()

        conf1 = conf1.permute(0, 2, 3, 1).contiguous()
        conf2 = conf2.permute(0, 2, 3, 1).contiguous()
        conf3 = conf3.permute(0, 2, 3, 1).contiguous()
        conf4 = conf4.permute(0, 2, 3, 1).contiguous()
        conf5 = conf5.permute(0, 2, 3, 1).contiguous()
        conf6 = conf6.permute(0, 2, 3, 1).contiguous()

        loc = torch.cat([loc1.view(loc1.size(0), -1),
                         loc2.view(loc2.size(0), -1),
                         loc3.view(loc3.size(0), -1),
                         loc4.view(loc4.size(0), -1),
                         loc5.view(loc5.size(0), -1),
                         loc6.view(loc6.size(0), -1)], 1)

        conf = torch.cat([conf1.view(conf1.size(0), -1),
                          conf2.view(conf2.size(0), -1),
                          conf3.view(conf3.size(0), -1),
                          conf4.view(conf4.size(0), -1),
                          conf5.view(conf5.size(0), -1),
                          conf6.view(conf6.size(0), -1)], 1)


        # output
        if self.phase == 'test':
            output = self.detect(loc.view(loc.size(0), -1, 4),
                                 self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                                 self.priors.type(type(x.data)))
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output


def build_shffulenetv2_ssd(phase, cfg, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    return ShuffleNetV2SSD(phase, size, num_classes, cfg)
