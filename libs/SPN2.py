import torch
import torch.nn as nn
from torchvision.models import vgg16
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn.functional as F
from libs.pytorch_spn.modules.gaterecurrent2dnoind import GateRecurrent2dnoind

class spn_block(nn.Module):
    def __init__(self, horizontal, reverse):
        super(spn_block, self).__init__()
        self.propagator = GateRecurrent2dnoind(horizontal,reverse)

    def forward(self,x,G1,G2,G3):
        sum_abs = G1.abs() + G2.abs() + G3.abs()
        sum_abs.data[sum_abs.data == 0] = 1e-6
        mask_need_norm = sum_abs.ge(1)
        mask_need_norm = mask_need_norm.float()
        G1_norm = torch.div(G1, sum_abs)
        G2_norm = torch.div(G2, sum_abs)
        G3_norm = torch.div(G3, sum_abs)

        G1 = torch.add(-mask_need_norm, 1) * G1 + mask_need_norm * G1_norm
        G2 = torch.add(-mask_need_norm, 1) * G2 + mask_need_norm * G2_norm
        G3 = torch.add(-mask_need_norm, 1) * G3 + mask_need_norm * G3_norm

        return self.propagator(x,G1,G2,G3)

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.conv1_1 = nn.Conv2d(3,64,3,padding = 1)
        self.conv1_2 = nn.Conv2d(64,64,3,padding = 1)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2,padding=1)
        self.conv2_1 = nn.Conv2d(64,128,3,padding = 1)
        self.conv2_2 = nn.Conv2d(128,128,3,padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2,padding=1)
        self.conv3_1 = nn.Conv2d(128,256,3,padding = 1)
        self.conv3_2 = nn.Conv2d(256,256,3,padding = 1)
        self.conv3_3 = nn.Conv2d(256,256,3,padding = 1)
        self.pool3 = nn.MaxPool2d(kernel_size = 3, stride = 2,padding=1)
        self.conv4_1 = nn.Conv2d(256,512,3,padding = 1)
        self.conv4_2 = nn.Conv2d(512,512,3,padding = 1)
        self.conv4_3 = nn.Conv2d(512,512,3,padding = 1)
        self.pool4 = nn.MaxPool2d(kernel_size = 3, stride = 2,padding=1)
        self.conv5_1 = nn.Conv2d(512,512,3,padding = 1)
        self.conv5_2 = nn.Conv2d(512,512,3,padding = 1)
        self.conv5_3 = nn.Conv2d(512,512,3,padding = 1)
        self.pool5 = nn.MaxPool2d(kernel_size = 3, stride = 2,padding=1)

    def forward(self,x):
        output = {}
        x = F.relu(self.conv1_1(x))
        x = self.conv1_2(x)
        x = self.pool1(F.relu(x))
        x = F.relu(self.conv2_1(x))
        x = self.pool2(F.relu(self.conv2_2(x)))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        output['conv3_3'] = self.conv3_3(x)
        x = self.pool3(F.relu(output['conv3_3']))
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        output['conv4_3'] = self.conv4_3(x)
        x = self.pool4(F.relu(output['conv4_3']))
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        output['conv5_3'] = self.conv5_3(x)
        output['pool5'] = self.pool5(F.relu(output['conv5_3']))
        return output

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layer0 = nn.Conv2d(512,512,1,1,0) # edge_conv5
        self.layer1 = nn.Upsample(scale_factor=2,mode='bilinear')
        self.layer2 = nn.Sequential(nn.Conv2d(512,512,3,1,1), # edge_conv6
                                    nn.ReLU(inplace=True))
        # sum : conv5_3 -> 28
        self.layer3 = nn.Upsample(scale_factor=2,mode='bilinear')
        self.layer4 = nn.Sequential(nn.Conv2d(512,512,3,1,1), # edge_conv7
                                    nn.ReLU(inplace=True))
        # sum: conv4_3 -> 21
        self.layer5 = nn.Upsample(scale_factor=2,mode='bilinear')
        self.layer6 = nn.Sequential(nn.Conv2d(512,256,3,1,1), # edge_conv8
                                    nn.ELU(inplace=True))
        # sum: conv3_3 -> 14
        self.layer7 = nn.Upsample(scale_factor=2,mode='bilinear')
        self.layer8 = nn.Conv2d(256,384,3,1,1)

    def forward(self,encode_feature):
        output = {}
        output['0'] = self.layer0(encode_feature['pool5'])
        output['1'] = self.layer1(output['0'])
        output['2'] = self.layer2(output['1'])

        output['2res'] = output['2'] + encode_feature['conv5_3']
        output['3'] = self.layer2(output['2res'])
        output['4'] = self.layer3(output['3'])

        output['4res'] = output['4'] + encode_feature['conv4_3']
        output['5'] = self.layer5(output['4res'])
        output['6'] = self.layer6(output['5'])

        output['6res'] = output['6'] + encode_feature['conv3_3']
        output['7'] = self.layer7(output['6res'])
        output['8'] = self.layer8(output['7'])

        return output['8']


class SPN(nn.Module):
    def __init__(self,vggPath=None,dataset='VOC'):
        super(SPN,self).__init__()
        if(dataset == 'VOC'):
            classNum = 21
        else:
            classNum = 19
            
        # conv for mask
        self.mask_conv = nn.Conv2d(classNum,32,4,2,1)

        # guidance network
        self.encoder = VGG()
        if not (vggPath is None):
            self.encoder.load_state_dict(torch.load(vggPath))
        self.decoder = Decoder()

        # spn blocks
        self.left_right_1 = spn_block(True,False)
        self.right_left_1 = spn_block(True,True)
        self.top_down_1 = spn_block(False, False)
        self.down_top_1 = spn_block(False,True)

        self.left_right_2 = spn_block(True,False)
        self.right_left_2 = spn_block(True,True)
        self.top_down_2 = spn_block(False, False)
        self.down_top_2 = spn_block(False,True)

        # post upsample
        self.postupsample = nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear'),
                                          nn.Conv2d(32,64,3,1,1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(64,classNum,3,1,1))
        self.softmax = nn.LogSoftmax()

    def forward(self,x,rgb):
        # feature for mask
        X = self.mask_conv(x)

        # guidance
        features = self.encoder(rgb)
        guide = self.decoder(features)
        G = torch.split(guide,32,1)
        out1 = self.left_right_1(X,G[0],G[1],G[2])
        out2 = self.right_left_1(X,G[3],G[4],G[5])
        out3 = self.top_down_1(X,G[6],G[7],G[8])
        out4 = self.down_top_1(X,G[9],G[10],G[11])
        out_1 = torch.max(out1,out2)
        out_1 = torch.max(out_1,out3)
        out_1 = torch.max(out_1,out4)

        out5 = self.left_right_2(out_1,G[0],G[1],G[2])
        out6 = self.right_left_2(out_1,G[3],G[4],G[5])
        out7 = self.top_down_2(out_1,G[6],G[7],G[8])
        out8 = self.down_top_2(out_1,G[9],G[10],G[11])

        out = torch.max(out5,out6)
        out = torch.max(out,out7)
        out = torch.max(out,out8)

        # upsample
        out = self.postupsample(out)

        return self.softmax(out)

if __name__ == '__main__':
    x = Variable(torch.Tensor(1,21,128,128)).cuda()
    spn = SPN('/home/xtli/WEIGHTS/spn_voc/pytorch_deeplab_large_fov/deeplab_vgg_init.pth')
    spn = spn.cuda()
    rgb = Variable(torch.Tensor(1,3,128,128)).cuda()
    output = spn(x,rgb)
    print(output.size())
