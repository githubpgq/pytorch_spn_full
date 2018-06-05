import argparse
import os
from libs.DataLoader import VOCDataset, CityScapeDataset, CityScapeDatasetMS
import torch
import time
import scipy.io
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
from torch.autograd import Variable
from libs.utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataRoot",type=str,default="/Data/Data/pascal/VOCdevkit/VOC2012/",help='data path')
parser.add_argument("--FilePath",type=str,default="/Data/Data/pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/seg11valid.txt",help='path point to files containing all training stuff')
# parser.add_argument("--dataRoot",type=str,default="",help='data path')
# parser.add_argument("--FilePath",type=str,default="cityscapes/",help='path point to files containing all training stuff')

parser.add_argument("--batchSize",type=int,default=1,help='batch size')
parser.add_argument("--out_dir",type=str,default="samples/",help='save probability maps path')
parser.add_argument('--ms', action='store_true',help='Turn on multi-scale testing')
parser.add_argument("--valSize",type=int,default=1024,help='batch size')
parser.add_argument("--device",type=int,default=0,help='device id')
parser.add_argument("-c","--checkpoint_path",type=str,default='',help='path to checkpoint')
parser.add_argument("-s","--spn_num",type=int,default=2,help='1 or 2 spns')
parser.add_argument("--save",type=int,default=0,help='whether save probability maps')
parser.add_argument("--dataset",type=str,default="CityScape",help='VOC|CityScape')

################# PREPARATIONS #################
opt = parser.parse_args()
opt.cuda = torch.cuda.is_available()
torch.cuda.set_device(opt.device)
print(opt)

try:
    os.makedirs(os.path.join(opt.out_dir,'pm'))
except OSError:
    pass

if(opt.spn_num == 1):
    from libs.SPN import SPN
else:
    from libs.SPN2 import SPN

################# DATA #################
if(opt.dataset == 'VOC'):
    dataset = VOCDataset(opt.dataRoot,opt.FilePath,test=True,loadSize=opt.valSize)
else:
    if(opt.ms):
        dataset = CityScapeDatasetMS(opt.dataRoot,opt.FilePath,test=True,
                                     testSize=opt.valSize,testScales=[0.5,0.75,1])
    else:
        dataset = CityScapeDataset(opt.dataRoot,opt.FilePath,test=True,testSize=opt.valSize)

val_loader_ = torch.utils.data.DataLoader(dataset = dataset,
                                            batch_size = opt.batchSize,
                                            num_workers = 4,
                                            shuffle = False)

val_loader = iter(val_loader_)

###################### MODEL #####################
spn = SPN(dataset=opt.dataset)
checkpoints = torch.load(opt.checkpoint_path)
spn.load_state_dict(checkpoints['state_dict'])

################# GLOBAL VARIABLE #################
rgbV = Variable(torch.Tensor(opt.batchSize,3,opt.valSize,opt.valSize),requires_grad=False)
if(opt.dataset == 'VOC'):
    maskV = Variable(torch.Tensor(opt.batchSize,21,opt.valSize,opt.valSize),requires_grad=False)
else:
    maskV = Variable(torch.Tensor(opt.batchSize,19,opt.valSize,opt.valSize),requires_grad=False)
labelV = Variable(torch.LongTensor(opt.batchSize,opt.valSize,opt.valSize),requires_grad=False)

####################### GPU ######################
if(opt.cuda):
    cudnn.benchmark = True
    spn.cuda()
    rgbV = rgbV.cuda()
    maskV = maskV.cuda()
    labelV = labelV.cuda()

################# TESTING #################
def save_samples(input_var,target_var,output_var,tmp_save):
    scipy.io.savemat(tmp_save,dict(input=input_var.data.cpu().numpy(),active=output_var.data.cpu().numpy(),label=target_var.data.cpu().numpy()))

def pad_back(loadsize, image, h, w):
    #n, c, h, w = image.shape
    pad_w = int((loadsize-w)/2)
    pad_h = int((loadsize-h)/2)
    if image.ndim == 3:
        return image[:, pad_h:pad_h+h,pad_w:pad_w+w]
    else:
        return image[:,:, pad_h:pad_h+h,pad_w:pad_w+w]


def test(num_classes=21):
    hist = np.zeros((num_classes, num_classes))
    hist_base = np.zeros((num_classes, num_classes))
    counter = 1
    for (rgb, mask, label, rgb_name) in iter(val_loader_):

        labelV.data.resize_(label.size()).copy_(label)

        if not opt.ms:
            _, coar = torch.max(mask,1)
            coar = coar.cpu().numpy()
            rgbV.data.resize_(rgb.size()).copy_(rgb)
            maskV.data.resize_(mask.size()).copy_(mask)

            #coar = pad_back(opt.valSize, coar, label.size()[1], label.size()[2])
            #print(label.shape, coar.shape)

            # forward
            final = spn(maskV,rgbV)
            final = final.exp().cpu().data.numpy()
            #final = pad_back(opt.valSize, final, label.size()[1], label.size()[2])
            valid_lbs = np.unique(coar[coar<22])
            final_correct = np.zeros_like(final)
            final_correct[:,valid_lbs] = final[:,valid_lbs]
            pred = final_correct.argmax(1)
        else:
            b,c,h,w = rgb[-1].size()
            finals = []
            for i in range(len(rgb)):
                _, coar = torch.max(mask[i],1)
                coar = coar.cpu().numpy()
                r = rgb[i]
                m = mask[i]
                rgbV.data.resize_(r.size()).copy_(r)
                maskV.data.resize_(m.size()).copy_(m)
                fi = spn(maskV,rgbV)
                fi = fi.exp().cpu().data.numpy()
                valid_lbs = np.unique(coar[coar<22])
                final_correct = np.zeros_like(fi)
                final_correct[:,valid_lbs] = fi[:,valid_lbs]
                finals.append(final_correct)


            result = finals[-1]
            for f in finals:
                if(f.shape[2] < 1024):
                    final = resize_4d_tensor(f)
                    result += final
            pred = result.argmax(axis=1)

        if(opt.save == 1):
            save_pm(maskV,Variable(torch.from_numpy(final_correct),requires_grad=False),labelV,os.path.join(opt.out_dir,'pm'),name=rgb_name,dataset="CityScapes")



        label = label.numpy()

        hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
        hist_base += fast_hist(coar.flatten(), label.flatten(), num_classes)
        mAP = round(np.nanmean(per_class_iu(hist)) * 100, 2)
        mAP_base = round(np.nanmean(per_class_iu(hist_base)) * 100, 2)
        print('===> Sample %d ===> mAP : %.3f, %.3f' % (counter, mAP, mAP_base))
        #print('===> Sample %d ===> mAP : %.3f' % (counter, mAP_base))


        counter += 1
    ious = per_class_iu(hist) * 100
    ious_base = per_class_iu(hist_base) * 100
    return round(np.nanmean(ious), 2), round(np.nanmean(ious_base), 2)


spn.eval()

if(opt.dataset == 'VOC'):
    print('===> mAP: %f, %f' % test())
else:
    print('===> mAP: %f, %f' % test(num_classes = 19))
