import torch
import numpy as np
from torch import nn
# test


class VGGBackbone(torch.nn.Module):
    def __init__(self,config,input_channel,use_bn,device='cpu'):
        super(VGGBackbone,self).__init__()
        self.device = device
        channels = config['channels']
        features = []
        self.use_bn = use_bn
        self.input_channel = input_channel

        for i, out_channels in channels:
            features += [
                nn.Conv2d(self.input_channel, out_channels, kernel_size=3, stride = 1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels) if self.use_bn else nn.Identity()
            ]
            
            if i == 1 or i == 3 or i == 5:
                features.append(nn.MaxPool2d(kernel_size=2, stride=2))

            self.input_channel = out_channels

        self.features = nn.Sequential(*features)

    def forward(self,x):
        out = self.features(x)
        return out



if __name__=="__main__":
    t = torch.randint(0,255,(1,1,240,320),dtype=torch.float32)
    
    config = {'backbone_type': 'VGG',
              'vgg': {'channels': [64, 64, 64, 64, 128, 128, 128, 128],
                      'convKernelSize': 3},
                      'det_head': {'feat_in_dim': 128},
                                   'des_head': {'feat_in_dim': 128,
                                                'feat_out_dim': 256}
              }
    net = VGGBackbone(config['vgg'],input_channel=1,use_bn=True, device = 'mps')
    print(net)
    out = net(t)
    print(out.shape)
