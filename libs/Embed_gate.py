import torch
import torch.nn as nn
from torch.autograd import Variable

"""
Embeded gate for SPN for multi-structure data segmentation
"""

def distance(x1, x2, mode = "dot"):
    if mode in "dot":
        return torch.mul(x1, x2)
    else:
        return -torch.pow((x1-x2), 2)

class Embeded_gate(nn.Module):
    """
    A factorized edge computation for SPN
    input: an n*c*h*w feature map
    output: gate maps given different directions for (a) x_pos, (b) x_neg, (c) y_pos, (d) y_neg, controlled by horizontal and reverse
    """
    def __init__(self, horizontal, reverse, mode = "dot"):
        super(Embeded_gate, self).__init__()
        # self.g1_r = nn.Parameter(torch.zeros_like(fea_).cuda())
        self.register_parameter('g1_r', None)
        self.register_parameter('g2_r', None)
        self.register_parameter('g2_r', None)
        self.horizontal = horizontal
        self.reverse = reverse

    def forward(self, x):
        is self.g1_r is None:
            self.g1_r = nn.Parameter(torch.zeros_like(x).cuda())
            self.g2_r = nn.Parameter(torch.zeros_like(x).cuda())
            self.g3_r = nn.Parameter(torch.zeros_like(x).cuda())
        if self.horizontal:
            if not self.reverse:
                self.g1_r[:,:,:-1,:-1] = x[:,:,1:,1:]
                self.g1_r = distance(self.g1_r, x, mode)
                self.g2_r[:,:,:,:-1] = x[:,:,:,1:]
                self.g2_r = distance(self.g2_r, x, mode)
                self.g3_r[:,:,1:,:-1] = x[:,:,:-1,1:]
                self.g3_r = distance(self.g3_r, x, mode)
            else:
                self.g1_r[:,:,:-1,1:] = x[:,:,1:,:-1]
                self.g1_r = distance(self.g1_r, x, mode)
                self.g2_r[:,:,:,1:] = x[:,:,:,:-1]
                self.g2_r = distance(self.g2_r, x, mode)
                self.g3_r[:,:,1:,1:] = x[:,:,:-1,:-1]
                self.g3_r = distance(self.g3_r, x, mode)
        else:
            self.g1_r = self.g1_r.transpose(3,2)
            self.g2_r = self.g2_r.transpose(3,2)
            self.g3_r = self.g3_r.transpose(3,2)
            xt = x.transpose(3,2)
            if not self.reverse:
                self.g1_r[:,:,:-1,:-1] = xt[:,:,1:,1:]
                self.g1_r = distance(self.g1_r, xt, mode)
                self.g2_r[:,:,:,:-1] = x[:,:,:,1:]
                self.g2_r = distance(self.g2_r, xt, mode)
                self.g3_r[:,:,1:,:-1] = x[:,:,:-1,1:]
                self.g3_r = distance(self.g3_r, xt, mode)
            else:
                self.g1_r[:,:,:-1,1:] = x[:,:,1:,:-1]
                self.g1_r = distance(self.g1_r, xt, mode)
                self.g2_r[:,:,:,1:] = x[:,:,:,:-1]
                self.g2_r = distance(self.g2_r, xt, mode)
                self.g3_r[:,:,1:,1:] = x[:,:,:-1,:-1]
                self.g3_r = distance(self.g3_r, xt, mode)
            self.g1_r = self.g1_r.transpose(3,2)
            self.g2_r = self.g2_r.transpose(3,2)
            self.g3_r = self.g3_r.transpose(3,2)

        return self.g1_r, self.g2_r, self.g3_r
