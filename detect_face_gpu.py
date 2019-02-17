from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime,time,math
import argparse
import numpy as np

import net_s3fd
from bbox import *

net = getattr(net_s3fd,'s3fd')()
net.load_state_dict(torch.load("s3fd_convert.pth"))
net.cuda()
def detect(img):
    img = img - np.array([104,117,123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,)+img.shape)

    img = Variable(torch.from_numpy(img).float(),volatile=True).cuda()
    BB,CC,HH,WW = img.size()
    olist = net(img)
    bboxlist = []
    for i in range(int(len(olist)/2)): olist[i*2] = F.softmax(olist[i*2])
    for i in range(int(len(olist)/2)):
        ocls,oreg = olist[i*2].data.cpu(),olist[i*2+1].data.cpu()
        FB,FC,FH,FW = ocls.size() # feature map size
        stride = 2**(i+2)    # 4,8,16,32,64,128
        anchor = stride*4
        for Findex in range(FH*FW):
            windex,hindex = Findex%FW,Findex//FW
            axc,ayc = stride/2+windex*stride,stride/2+hindex*stride
            score = ocls[0,1,hindex,windex]
            loc = oreg[0,:,hindex,windex].contiguous().view(1,4)
            if score<0.05: continue
            priors = torch.Tensor([[axc/1.0,ayc/1.0,stride*4/1.0,stride*4/1.0]])
            variances = [0.1,0.2]
            box = decode(loc,priors,variances)
            x1,y1,x2,y2 = box[0]*1.0
            # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            if score<0.5: continue
            bboxlist.append([x1,y1,x2,y2,score])
    bboxlist = np.array(bboxlist)
    if 0==len(bboxlist): bboxlist=np.zeros((1, 5))
    keep = nms(bboxlist,0.3)
    bboxlist = bboxlist[keep,:]
    return bboxlist
