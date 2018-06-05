import torch
import torch.nn as nn
from torchvision.models import vgg16
from torch.autograd import Variable
from collections import OrderedDict
from libs.pytorch_spn.modules.gaterecurrent2dnoind import GateRecurrent2dnoind
from libs.SPNModule import spn_block


class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        vgg = vgg16()
        vgg.load_state_dict(torch.load('vgg_models/vgg16-397923af.pth'))
        vgg = vgg.features
        self.layer0 = vgg._modules['0']
        self.layer1 = vgg._modules['1']
        self.layer2 = vgg._modules['2']
        self.layer3 = vgg._modules['3']
        self.layer4 = vgg._modules['4']
        self.layer5 = vgg._modules['5']
        self.layer6 = vgg._modules['6']
        self.layer7 = vgg._modules['7']
        self.layer8 = vgg._modules['8']
        self.layer9 = vgg._modules['9']
        self.layer10 = vgg._modules['10']
        self.layer11 = vgg._modules['11']
        self.layer12 = vgg._modules['12']
        self.layer13 = vgg._modules['13']
        self.layer14 = vgg._modules['14']
        self.layer15 = vgg._modules['15']
        self.layer16 = vgg._modules['16']
        self.layer17 = vgg._modules['17']
        self.layer18 = vgg._modules['18']
        self.layer19 = vgg._modules['19']
        self.layer20 = vgg._modules['20']
        self.layer21 = vgg._modules['21']
        self.layer22 = vgg._modules['22']
        self.layer23 = vgg._modules['23']
        self.layer24 = vgg._modules['24']
        self.layer25 = vgg._modules['25']
        self.layer26 = vgg._modules['26']
        self.layer27 = vgg._modules['27']
        self.layer28 = vgg._modules['28']
        self.layer29 = vgg._modules['29']
        self.layer30 = vgg._modules['30']

    def forward(self,x):
        output = {}
        output['-1'] = x
        for i in range(0,31):
            layer = getattr(self,'layer'+str(i))
            output[str(i)] = layer(output[str(i-1)])
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
        self.layer8 = nn.Conv2d(256,32,3,1,1)

    def forward(self,encode_feature):
        output = {}
        output['0'] = self.layer0(encode_feature['30'])
        output['1'] = self.layer1(output['0'])
        output['2'] = self.layer2(output['1'])

        output['2res'] = output['2'] + encode_feature['28']
        output['3'] = self.layer2(output['2res'])
        output['4'] = self.layer3(output['3'])

        output['4res'] = output['4'] + encode_feature['21']
        output['5'] = self.layer5(output['4res'])
        output['6'] = self.layer6(output['5'])

        output['6res'] = output['6'] + encode_feature['14']
        output['7'] = self.layer7(output['6res'])
        output['8'] = self.layer8(output['7'])

        return output['8']


class SPN(nn.Module):
    def __init__(self):
        super(SPN,self).__init__()
        # conv for mask
        self.mask_conv = nn.Conv2d(19,32,4,2,1)

        # guidance network
        self.encoder = VGG()
        self.decoder = Decoder()

        # spn blocks
        self.left_right = spn_block(True,False)
        self.right_left = spn_block(True,True)
        self.top_down = spn_block(False, False)
        self.down_top = spn_block(False,True)

        # post upsample
        self.postupsample = nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear'),
                                          nn.Conv2d(32,64,3,1,1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(64,19,3,1,1))
        self.softmax = nn.LogSoftmax()

    def forward(self,x,rgb):
        # feature for mask
        X = self.mask_conv(x)

        # guidance
        features = self.encoder(rgb)
        guide = self.decoder(features)
        # G = torch.split(guide,32,1)
        out1 = self.left_right(X,guide)
        out2 = self.right_left(X,guide)
        out3 = self.top_down(X,guide)
        out4 = self.down_top(X,guide)
        out = torch.max(out1,out2)
        out = torch.max(out,out3)
        out = torch.max(out,out4)

        # upsample
        outu = self.postupsample(out)

        return self.softmax(outu)

        # return guide

# if __name__ == '__main__':
#     x = Variable(torch.Tensor(1,3,224,224)).cuda()
#     spn = SPN()
#     spn = spn.cuda()
#     rgb = Variable(torch.Tensor(1,3,224,224)).cuda()
#     output = spn(x,rgb)
