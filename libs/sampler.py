import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

class sampler(nn.Module):
    """produce a mask according to current label"""
    def __init__(self, valid_cls):
        super(sampler, self).__init__()
        # valid_cls = np.array([1,3,4,5,6,7,9,11,12,13,14,15,16,17,18])
        self.valid_cls = valid_cls

    def forward(self, label):
        lab = label.data.numpy()
        nidx = np.unique(lab[lab<19])
        vidx = np.intersect1d(np.unique(lab[lab<19]), self.valid_cls)
        if len(nidx) > len(vidx):
            mask = np.zeros(lab.shape)
            lab_r = np.repeat(np.expand_dims(lab,0),len(vidx),axis=0)
            vidx_r = np.expand_dims(np.expand_dims(np.array(vidx),1),2)
            scls_ = np.sum((lab_r==vidx_r)*1,axis=0)

            vidx_bar = np.setdiff1d(nidx,vidx,assume_unique=True)
            lab_bar_r = np.repeat(np.expand_dims(lab,0),len(vidx_bar),axis=0)
            vidx_bar_r = np.expand_dims(np.expand_dims(np.array(vidx_bar),1),2)
            lcls_ = np.sum((lab_bar_r == vidx_bar_r) * 1, axis = 0)
            mask[scls_] = 1
            if len(scls_) * 8 < len(lcls_):
                lcls_ = np.random.permutation(lcls_)
                lcls_select = lcls_[:len(scls_) * 8]
                mask[lcls_select] = 1
            else:
                mask[lcls_] = 1
        else:
             mask = np.ones(lab.shape)
        maskV = Variable(torch.FloatTensor(mask), requires_grad = False)
        return maskV
