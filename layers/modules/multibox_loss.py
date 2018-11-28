# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        # priorbox： 8732，4
        # loc_data： N， 8732， 4
        # conf_data：N， 8732， 21
        loc_data, conf_data, priors = predictions
        # N, batch size
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        # num_priors: 8732
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        #, N,8732,4
        loc_t = torch.Tensor(num, num_priors, 4)
        # N,8732, 21
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            # 根据交并比，将真值框与Priorbox匹配起来
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        # loc_t（真值）放置的是8732个 priorbox与其对应的truthbox编码过的bbox： N，8732，4
        # conf_t（真值）放置的是label (里面除0标签外，其他标签加了个1）： （N，8732）
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        # pos就是标签不为背景的地方，torch.uint8，背景的地方为0，有目标的地方为1： (N,8732)
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        # pos.unsqueeze(pos.dim()): 扩充维为： N, 8732, 1 ，然后再展成 N, 8732, 4
        # 展开过程中，会把一行的1，展成1，1，1，1
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        # 所有对应有真值框的位置(prior box)，都计算出一个位置 loss
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')


        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # log_sum_exp可以看成是softmax,输出为，N*8732, 1
        # batch_conf的sahep是： N*8732, 21
        # conf_t的shape是： N,8732
        # 这个计算是什么意思？ -> 应该是计算loss的，但相应的公式没有找到
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        # loss_c的shape：N，8732


        # Hard Negative Mining
        # 把正样本对应的Loss位置，置成0（取负样本时，不会取到正样本）
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)

        # 将loss_c按降序排序，loss_idx就是loss的索引，
        # 此时，取loss_idx的前 X个（num_pos个的3倍，num_neg个）索引对应的样本，
        # 作为参与计算的负样本即可
        _, loss_idx = loss_c.sort(1, descending=True)

        # 将loss_idx按升序排序，即将loss_idx还原成0,1,...,8732的形式，
        # 将loss_idx还原成正常顺序后，其排序索引idx_rank则记录了loss值的排序位置
        # idx_rank中，值越小，说明该位置样本的loss值越大
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)

        # 将idx_rank中，小于num_neg的置为1，其余置为0，即取了loss_idx的前num_neg个样本
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum().float()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
