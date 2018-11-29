from __future__ import print_function
import torch
import cv2
import time
from pylab import plt
from collections import OrderedDict

# weights = "../weights/ssd300_mAP_77.43_v2.pth"
# weights = "../weights/VOC.pth"
weights = "../models/shufflenetv2_ssd_detach_20181126_iter_60000.pkl"
use_cuda = True

device = torch.device('cuda:0' if use_cuda else 'cpu')

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from data import BaseTransform, VOC_CLASSES as labelmap
from shufflenetv2_ssd_detach import *


# ssd base net
net = ShuffleNetV2SSD_Detach("test", 300, 21, 1)
w = torch.load(weights)
nw = OrderedDict()
for k, v in w.items():
    name = k[7:]
    nw[name] = v
net.load_state_dict(nw)
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

net.to(device)
net.eval()

# priorbox
from layers.functions import prior_box, detection
from data.config import *
cfg = voc_shufflenetv2
# priorbox
net_priorbox = PriorBox(cfg)
with torch.no_grad():
    priorboxes = net_priorbox.forward()
priorboxes = priorboxes.to(device)

# criterion
out_layer = Detect(21, 0, 200, 0.01, 0.45)
out_layer.to(device)
out_layer.eval()




def predict(frame):
    height, width = frame.shape[:2]
    x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
    x = x.unsqueeze(0).to(device)
    begin = time.time()
    with torch.no_grad():
        y = net(x)  # forward pass
        loc, conf = y
        y = out_layer(loc, conf, priorboxes)
    end = time.time()
    print("preidct time: {} ms".format((end-begin) * 1000))
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([width, height, width, height])
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            cv2.rectangle(frame,
                          (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])),
                          COLORS[i % 3], 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                        FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame

img_path = "/home/beichen2012/dataset/VOCdevkit/VOC2012/JPEGImages/2012_003937.jpg"
img = cv2.imread(img_path, 1)

# for i in range(0,5):
#     frame = predict(img)


begin = time.time()
img = predict(img)
end = time.time()
print("time cost: {} ms".format((end-begin) * 1000.0))
plt.figure()
plt.imshow(img[:,:,::-1])
plt.show()

