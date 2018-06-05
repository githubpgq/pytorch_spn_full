import os
import cv2
import torch
import time
import torch.utils.data as data
from PIL import Image
import numpy as np
import random
import scipy.io
import libs.data_transforms as transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

class VOCDataset(data.Dataset):
    def __init__(self,dataRoot,nameFile,loadSize=None,scale=1,patch_num=1,cropSize=None,rotate=None,test=False):
        super(VOCDataset,self).__init__()
        self.dataRoot = dataRoot

        self.name_list = [line.strip() for line in open(nameFile,'r')]
        self.cropSize = cropSize
        self.test = test
        self.patch_num = patch_num

        mean = [104.00699, 116.66877, 122.67892] #BGR
        # TODO: what's the std for caffe VGG
        #std = [0.229, 0.224, 0.225]

        if(test):
            self.transform = transforms.Compose([
                             #transforms.Resize((loadSize,loadSize)),
                             transforms.pad(loadSize),
                             transforms.ToTensor(),
                             transforms.Normalize(mean)
                             ])
        else:
            self.transform = transforms.Compose([
                             #transforms.Resize(loadSize),
                             transforms.RandomScale(scale),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomRotation(rotate),
                             transforms.ConstrainedCrop(cropSize),
                             transforms.ToTensor(),
                             transforms.Normalize(mean)
                             ])

    def __getitem__(self,index):
        rgbPath = os.path.join(self.dataRoot,"JPEGImages",self.name_list[index]+".jpg")
        maskPath = os.path.join(self.dataRoot,"SegmentationProb",self.name_list[index]+".mat")
        labelPath = os.path.join(self.dataRoot,"SegmentationClass",self.name_list[index]+".png")

        # preprocess
        rgb = cv2.imread(rgbPath)
        mat = scipy.io.loadmat(maskPath)
        mask = mat['out']
        mask = mask.transpose(1,2,0)
        #label = cv2.imread(labelPath,0)
        # label is a special color map in VOC
        label = Image.open(labelPath)
        label = np.asarray(label)

        h,w = label.shape
        if(h < 128 or w < 128):
            if(h < w):
                newh = 128
                neww = int(newh*w/h)
            if(w < h):
                neww = 128
                newh = int(neww*h/w)
            rgb, mask, label = transforms.Resize((newh,neww))(rgb, mask, label)

        if(self.test):
            rgb, mask, label = self.transform(rgb, mask, label)
            #mask = torch.exp(mask)
            return rgb, mask, label ,self.name_list[index]
        else:
            rgbs = torch.Tensor(self.patch_num,3,self.cropSize,self.cropSize)
            masks = torch.Tensor(self.patch_num,21,self.cropSize,self.cropSize)
            labels = torch.Tensor(self.patch_num,self.cropSize,self.cropSize)
            for i in range(self.patch_num):
                rgb_patch, mask_patch, label_patch = self.transform(rgb,mask,label)
                rgbs[i,:,:,:] = rgb_patch
                masks[i,:,:,:] = mask_patch
                labels[i,:,:] = label_patch
            #masks = torch.exp(masks)
            return rgbs, masks, labels ,self.name_list[index]
        

    def __len__(self):
        return len(self.name_list)

class CityScapeDataset(data.Dataset):
    def __init__(self,dataRoot,FilePath,scale=None,patch_num=8,cropSize=None,rotate=None,test=False,testSize=512):
        super(CityScapeDataset,self).__init__()
        self.dataRoot = dataRoot

        if not test:
            rgbPath = os.path.join(FilePath,'train_images.txt')
            maskPath = os.path.join(FilePath,'train_masks.txt')
            labelPath = os.path.join(FilePath,'train_labels.txt')
        else:
            rgbPath = os.path.join(FilePath,'val_images.txt')
            maskPath = os.path.join(FilePath,'val_masks.txt')
            labelPath = os.path.join(FilePath,'val_labels.txt')

        self.rgb_list = [line.strip() for line in open(rgbPath,'r')]
        self.mask_list = [line.strip() for line in open(maskPath,'r')]
        self.label_list = [line.strip() for line in open(labelPath,'r')]
        self.cropSize = cropSize
        self.patch_num = patch_num
        self.test = test
        self.testSize = testSize

        mean = [104.00699, 116.66877, 122.67892] #BGR

        if(test):
            self.transform = transforms.Compose([
                             transforms.Resize(testSize),
                             transforms.ToTensor(),
                             transforms.Normalize(mean)
                             ])
        else:
            self.transform = transforms.Compose([
                             transforms.RandomScale(scale),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomRotation(rotate),
                             transforms.ConstrainedCropCityScapes(cropSize),
                             transforms.ToTensor(),
                             transforms.Normalize(mean)
                             ])

    def __getitem__(self,index):
        rgbPath = os.path.join(self.dataRoot,self.rgb_list[index])
        maskPath = os.path.join(self.dataRoot,self.mask_list[index])
        labelPath = os.path.join(self.dataRoot,self.label_list[index])
        
        if(self.test):
            #maskPath = maskPath.replace("drn_d_22_049_val_npz","drn_d_22_049_val_npz_"+str(self.testSize))
            maskPath = maskPath.replace("drn_d_22_117_val_npz","drn_d_22_117_val_npz_"+str(self.testSize))
        else:
            maskPath = maskPath.replace("drn_d_22_049_train_npz","drn_d_22_049_train_npz_1024")
        
        

        # preprocess
        rgb = cv2.imread(rgbPath)
        mask = np.load(maskPath) # 19 x 1024 x 2048
        mask = mask.transpose(1,2,0) # 1024 x 2048 x 19
        mask = np.exp(mask)
        label = cv2.imread(labelPath,0)

        if not self.test:
            rgbs = torch.Tensor(self.patch_num,3,self.cropSize,self.cropSize)
            masks = torch.Tensor(self.patch_num,19,self.cropSize,self.cropSize)
            labels = torch.Tensor(self.patch_num,self.cropSize,self.cropSize)
            for i in range(self.patch_num):
                rgb_patch, mask_patch, label_patch = self.transform(rgb, mask,label)
                rgbs[i,:,:,:] = rgb_patch
                masks[i,:,:,:] = mask_patch
                labels[i,:,:] = label_patch
            return rgbs, masks, labels, self.rgb_list[index]
        else:
            rgb, mask, label = self.transform(rgb, mask, label)
            return rgb, mask, label ,self.rgb_list[index]

    def __len__(self):
        return len(self.rgb_list)

class CityScapeDatasetMS(data.Dataset):
    def __init__(self,dataRoot,FilePath,scale=None,patch_num=8,cropSize=None,rotate=None,test=False,testSize=512,testScales=None):
        super(CityScapeDatasetMS,self).__init__()
        self.dataRoot = dataRoot

        if not test:
            rgbPath = os.path.join(FilePath,'train_images.txt')
            maskPath = os.path.join(FilePath,'train_masks.txt')
            labelPath = os.path.join(FilePath,'train_labels.txt')
        else:
            rgbPath = os.path.join(FilePath,'val_images.txt')
            maskPath = os.path.join(FilePath,'val_masks.txt')
            labelPath = os.path.join(FilePath,'val_labels.txt')

        self.rgb_list = [line.strip() for line in open(rgbPath,'r')]
        self.mask_list = [line.strip() for line in open(maskPath,'r')]
        self.label_list = [line.strip() for line in open(labelPath,'r')]
        self.cropSize = cropSize
        self.patch_num = patch_num
        self.test = test
        self.testSize = testSize
        self.testScales = testScales

        mean = [104.00699, 116.66877, 122.67892] #BGR

        if(test):
            self.transform = transforms.Compose([
                             transforms.Resize(testSize),
                             transforms.ToTensor(),
                             transforms.Normalize(mean)
                             ])
        else:
            self.transform = transforms.Compose([
                             transforms.RandomScale(scale),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomRotation(rotate),
                             transforms.ConstrainedCropCityScapes(cropSize),
                             transforms.ToTensor(),
                             transforms.Normalize(mean)
                             ])

    def __getitem__(self,index):
        rgbPath = os.path.join(self.dataRoot,self.rgb_list[index])
        maskPath = os.path.join(self.dataRoot,self.mask_list[index])
        labelPath = os.path.join(self.dataRoot,self.label_list[index])
        
        if(self.test):
            #maskPath = maskPath.replace("drn_d_22_049_val_npz","drn_d_22_049_val_npz_"+str(self.testSize))
            maskPath = maskPath.replace("drn_d_22_117_val_npz","drn_d_22_117_val_npz_"+str(self.testSize))
        else:
            maskPath = maskPath.replace("drn_d_22_049_train_npz","drn_d_22_049_train_npz_1024")
        
        

        # preprocess
        rgb = cv2.imread(rgbPath)
        mask = np.load(maskPath) # 19 x 1024 x 2048
        mask = mask.transpose(1,2,0) # 1024 x 2048 x 19
        mask = np.exp(mask)
        label = cv2.imread(labelPath,0)

        masks = []
        rgbs = []
        h,w,c = mask.shape
        for scale in self.testScales:
            newh = int(h*scale)
            neww = int(w*scale)
            resized_rgb, resized_mask, _ = transforms.Resize(newh)(rgb, mask, label)
            resized_rgb,resized_mask,_ = transforms.ToTensor()(resized_rgb,resized_mask,label)
            masks.append(resized_mask)
            rgbs.append(resized_rgb)
        rgb,mask,label = self.transform(rgb,mask,label)
        return rgbs, masks, label, self.rgb_list[index]

    def __len__(self):
        return len(self.rgb_list)