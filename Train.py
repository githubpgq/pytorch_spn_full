import argparse
import os
from libs.DataLoader import VOCDataset, CityScapeDataset
import torch
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
from torch.autograd import Variable
from libs.utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataRoot",type=str,default="/Data/Data/pascal/VOCdevkit/VOC2012/",help='data path')
parser.add_argument("--dataRoot",type=str,default="",help='data path')
parser.add_argument("--FilePath",type=str,default="/Data/Data/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt",help='path point to files containing all training stuff')
#parser.add_argument("--FilePath",type=str,default="cityscapes/",help='path point to files containing all training stuff')

parser.add_argument("--batchSize",type=int,default=1,help='batch size')
parser.add_argument("--niter",type=int,default=60000,help='training iterations')
parser.add_argument("--cropSize",type=int,default=128,help='crop size')
parser.add_argument("--rotate",type=int,default=10,help='degree to rotate training images')
parser.add_argument("--scale",type=int,default=2,help='random scale')
parser.add_argument("--lr",type=float,default=0.0001,help='learning rate')
parser.add_argument("--momentum",type=float,default=0.9,help='momentum')
parser.add_argument("--weight_decay",type=float,default=0.005,help='weight decay')
parser.add_argument("--evaluate_interval",type=int,default=50,help='#iterations to evaluate')
parser.add_argument("--save_interval",type=int,default=5000,help='#iterations to save models')
parser.add_argument("-c","--checkpoint_path",type=str,default="models/",help='checkpoints path')
parser.add_argument("--vggPath",type=str,default="weights/deeplab_vgg_init.pth",help='pre-trained vgg path')
parser.add_argument("--ave",type=int,default=1,help='loss norm')
parser.add_argument("--device",type=int,default=0,help='device id')
parser.add_argument("-s","--spn_num",type=int,default=2,help='1 or 2 spns')
parser.add_argument("--patch_num",type=int,default=10,help='how many patches per image')
parser.add_argument("--dataset",type=str,default="CityScape",help='VOC|CityScape')


################# PREPARATIONS #################
opt = parser.parse_args()
opt.cuda = torch.cuda.is_available()
torch.cuda.set_device(opt.device)
if(opt.spn_num == 1):
    print('using a single SPN')
    from libs.SPN import SPN
else:
    print('using 2 SPNs')
    from libs.SPN2 import SPN
print(opt)

try:
    os.makedirs(os.path.join(opt.checkpoint_path,'pm'))
except OSError:
    pass

################# DATA #################
if(opt.dataset == 'VOC'):
    dataset = VOCDataset(opt.dataRoot,opt.FilePath,cropSize=opt.cropSize,rotate=opt.rotate,scale=opt.scale,patch_num=opt.patch_num)
else:
    dataset = CityScapeDataset(opt.dataRoot,opt.FilePath,cropSize=opt.cropSize,rotate=opt.rotate,scale=opt.scale,patch_num=opt.patch_num)

train_loader_ = torch.utils.data.DataLoader(dataset = dataset,
                                            batch_size = opt.batchSize,
                                            num_workers = 6,
                                            shuffle = True,
                                            drop_last=True)

train_loader = iter(train_loader_)

###################### MODEL #####################
spn = SPN(opt.vggPath,opt.dataset)

################ LOSS & OPTIMIZER ################
criterion = nn.NLLLoss2d(ignore_index=255,size_average=False)
optimizer_encoder = torch.optim.SGD(spn.encoder.parameters(),
                            opt.lr*0.1,
                            momentum = opt.momentum,
                            weight_decay = opt.weight_decay)

if(opt.spn_num == 1):
    optimizer_rest_part = torch.optim.SGD([
                                {'params': spn.decoder.parameters()},
                                {'params': spn.mask_conv.parameters()},
                                {'params': spn.left_right.parameters()},
                                {'params': spn.right_left.parameters()},
                                {'params': spn.top_down.parameters()},
                                {'params': spn.down_top.parameters()},
                                {'params': spn.postupsample.parameters()}],
                                opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
else:
    optimizer_rest_part = torch.optim.SGD([
                                {'params': spn.decoder.parameters()},
                                {'params': spn.mask_conv.parameters()},
                                {'params': spn.left_right_1.parameters()},
                                {'params': spn.right_left_1.parameters()},
                                {'params': spn.top_down_1.parameters()},
                                {'params': spn.down_top_1.parameters()},
                                {'params': spn.left_right_2.parameters()},
                                {'params': spn.right_left_2.parameters()},
                                {'params': spn.top_down_2.parameters()},
                                {'params': spn.down_top_2.parameters()},
                                {'params': spn.postupsample.parameters()}],
                                opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)


################# GLOBAL VARIABLE #################
rgbV = Variable(torch.Tensor(opt.batchSize*opt.patch_num,3,opt.cropSize,opt.cropSize))
if(opt.dataset == 'VOC'):
    maskV = Variable(torch.Tensor(opt.batchSize*opt.patch_num,21,opt.cropSize,opt.cropSize))
else:
    maskV = Variable(torch.Tensor(opt.batchSize*opt.patch_num,19,opt.cropSize,opt.cropSize))
labelV = Variable(torch.LongTensor(opt.batchSize*opt.patch_num,opt.cropSize,opt.cropSize))

####################### GPU ######################
if(opt.cuda):
    cudnn.benchmark = True
    spn.cuda()
    criterion.cuda()
    rgbV = rgbV.cuda()
    maskV = maskV.cuda()
    labelV = labelV.cuda()

################# TRAINING #################
def adjust_learning_rate(optimizer_encoder, optimizer_rest_part,iteration):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_encoder = opt.lr * 0.1 * (1 - iteration * 1.0 / opt.niter) ** 0.5
    lr_rest_part = opt.lr * (1 - iteration * 1.0 / opt.niter) ** 0.5

    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = lr_encoder

    for param_group in optimizer_rest_part.param_groups:
        param_group['lr'] = lr_rest_part
    return lr_encoder, lr_rest_part

def save_checkpoint(state, is_best, iteration):
    filename = os.path.join(opt.checkpoint_path, 'checkpoint_iteration_%d.pth'%iteration)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def loadData(rgbs, masks, labels):
    patch_num = opt.patch_num
    for i in range(opt.batchSize):
        rgbV.data[i*patch_num:i*patch_num+patch_num] = rgbs[i,:,:,:,:].squeeze().clone()
        maskV.data[i*patch_num:i*patch_num+patch_num] = masks[i,:,:,:,:].squeeze().clone()
        labelV.data[i*patch_num:i*patch_num+patch_num] = labels[i,:,:,:].squeeze().clone()
    #return rgbV, maskV, labelV

#spn = torch.nn.DataParallel(spn)
spn.train()
batch_time = AverageMeter()
losses = AverageMeter()
scores = AverageMeter()
for iteration in range(1,opt.niter+1):
    start_time = time.time()
    optimizer_encoder.zero_grad()
    optimizer_rest_part.zero_grad()
    try:
        rgb,mask,label,_ = train_loader.next()
    except StopIteration:
        print('StopIteration Encountered')
        train_loader = iter(train_loader_)
        rgb,mask,label,_ = train_loader.next()
        batch_time = AverageMeter()
        losses = AverageMeter()
        scores = AverageMeter()

    #rgbV.data.copy_(rgb.squeeze())
    #maskV.data.copy_(mask.squeeze())
    #labelV.data.copy_(label.squeeze())
    loadData(rgb, mask, label)

    # forward
    predict = spn(maskV,rgbV)
    loss = criterion(predict,labelV) / (opt.batchSize * opt.patch_num)
    # backward
    loss.backward()
    nn.utils.clip_grad_norm(spn.parameters(), 1000)
    # clip_grad_value_(spn.parameters(), 1000)
    optimizer_encoder.step()
    optimizer_rest_part.step()
    end_time = time.time()

    # logging
    score = accuracy(predict,labelV)
    losses.update(loss.data[0], opt.batchSize)
    scores.update(score, opt.batchSize)
    batch_time.update((end_time-start_time))
    print("Iteration: [%d/%d] Loss: %.4f (%.4f) Score: %.4f(%.4f) Time: %.4f(%.4f) Encoder LR: %.6f Rest LR: %.6f."
    %(iteration,opt.niter,loss.data[0],losses.avg,score,scores.avg,(end_time-start_time),batch_time.avg,
     optimizer_encoder.param_groups[0]['lr'], optimizer_rest_part.param_groups[0]['lr']))

    # learning rate decay
    adjust_learning_rate(optimizer_encoder,optimizer_rest_part,iteration)

    # maybe save models
    if(iteration % opt.save_interval == 0):
        save_checkpoint({'iteration': iteration,
                         'state_dict': spn.state_dict()}, False, iteration)

    if(iteration % opt.evaluate_interval == 0):
        save_pm(maskV[0:10],predict[0:10],labelV[0:10],os.path.join(opt.checkpoint_path,'pm'))
