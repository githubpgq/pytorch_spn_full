from __future__ import division
import torch
import random
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import numbers
import os
import cv2
import math
import threading
from torchvision import datasets, transforms
import torchvision.utils as vutils

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 255]
    correct = correct.view(-1)
    score = correct.float().sum(0).mul(100.0 / correct.size(0))
    return score.data[0]

def save_colorful_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    CITYSCAPE_PALETTE = np.asarray([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [0, 0, 0]], dtype=np.uint8)
    predictions[predictions == 255] = 19
    im = Image.fromarray(CITYSCAPE_PALETTE[predictions.squeeze()])
    fn = os.path.join(output_dir, filenames[:-4] + '.png')
    out_dir = os.path.split(fn)[0]
    if not os.path.exists(out_dir):
       os.makedirs(out_dir)
    im.save(fn)

def resize_4d_tensor(f, height=1024, width=2048):
    f = f.squeeze()
    channel = f.shape[0]
    final = np.zeros((1,channel,height,width))
    for ii in range(channel):
        fii = f[ii,:,:]
        fii = cv2.resize(fii, (width,height),interpolation=cv2.INTER_LINEAR)
        final[0,ii,:,:] = fii
    return final

def tensor2blocks(t,k,s):
    """
    input:
    - b x c x h x w tensor
    - k: kernel size
    - s: strid
    output:
    - (floor((h-k+1)/s+1))x(floor((w-k+1)/s+1)) blocks each with size k x k
    """
    b,c,h,w = t.size()
    blocks = []
    bh = int(math.floor((h-k+1)/s+1))
    bw = int(math.floor((w-k+1)/s+1))
    for i in range(bh):
        for j in range(bw):
            bi = t.data[:,:,i*s:(i*s+k),j*s:(j*s+k)].clone()
            blocks.append(bi)
    return blocks

def blocks2tensor(bs,k,s,h,w):
    """
    input:
    - bs: a list of blocks
    - k: kernel size
    - s: stride
    - h,w: size of original tensor
    """
    b,c,k,k = bs[0].size()
    bh = int(math.floor((h-k+1)/s+1))
    bw = int(math.floor((w-k+1)/s+1))
    t = torch.zeros(b,c,h,w)
    # counter calculates how much overlap one posistion has
    counter = torch.zeros(b,c,h,w)
    for i in range(bh):
        for j in range(bw):
            t[:,:,i*s:(i*s+k),j*s:(j*s+k)] += bs[i*bh+j]
            counter[:,:,i*s:(i*s+k),j*s:(j*s+k)] += torch.ones(b,c,k,k)
    return t/counter

def save_pm(input,prediction,gt,path,name=None,dataset="VOC"):
    """
    input: b x 21 x h x w
    prediction: b x 21 x h x w
    gt: b x h x w
    """
    b,c,h,w = input.size()
    input = input.data.cpu()
    prediction = prediction.data.cpu()
    gt = gt.data.cpu()
    for i in range(b):
        # for each mask, we output 21 probability maps
        input_i = input[i].squeeze()
        prediction_i = prediction[i].squeeze()
        prediction_i = torch.exp(prediction_i)
        gt_i = gt[i].squeeze()
        
        
        
        if not (name is None):
            if(dataset == 'VOC'):
                name_base = os.path.join(path,name[i])
            else:
                name_base = os.path.join(path,name[i].split('/')[-1][:-4])
            vutils.save_image(input_i.unsqueeze(1),name_base+'_input.png',normalize=True,scale_each=True,nrow=7)
            vutils.save_image(prediction_i.unsqueeze(1),name_base+'prediction.png',normalize=True,scale_each=True,nrow=7)
            vutils.save_image(gt_i,name_base+'_gt.png',normalize=True,scale_each=True)
        else:
            vutils.save_image(input_i.unsqueeze(1),os.path.join(path,'%d_input.png'%i),normalize=True,scale_each=True,nrow=7)
            vutils.save_image(prediction_i.unsqueeze(1),os.path.join(path,'%d_prediction.png'%i),normalize=True,scale_each=True,nrow=7)
            vutils.save_image(gt_i,os.path.join(path,'%d_gt.png'%i),normalize=True,scale_each=True)
        

def clip_grad_value_(parameters, clip_value):
    r"""Clips gradient of an iterable of parameters at specified value.

    Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor]): an iterable of Tensors that will have
            gradients normalized
        clip_value (float or int): maximum allowed value of the gradients
            The gradients are clipped in the range [-clip_value, clip_value]
    """
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad.data.clamp_(min=-clip_value, max=clip_value)