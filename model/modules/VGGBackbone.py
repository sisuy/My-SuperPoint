import torch
import numpy as np

class VGGBackbone(torch.nn.Module):
    def __init__(self,config,input_channel = 1, device = 'cpu'):
        super(VGGBackbone,self).__init__()
        self.device = device
        channels = config['channels']

        self.block1_1 = torch.nn.Sequential(
                torch.nn.Conv2d(input_channel,channels[0],kernel_size = 3,stride = 1,padding = 1),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(channels[0])
                )

        self.block1_2 = torch.nn.Sequential(
                torch.nn.Conv2d(channels[0],channels[1],kernel_size = 3,stride = 1,padding = 1),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(channels[1]),
                torch.nn.MaxPool2d(kernel_size=2,stride=2)
                )

        self.block2_1 = torch.nn.Sequential(
                torch.nn.Conv2d(channels[1],channels[2],kernel_size = 3,stride = 1,padding = 1),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(channels[2]),
                )

        self.block2_2 = torch.nn.Sequential(
                torch.nn.Conv2d(channels[2],channels[3],kernel_size = 3,stride = 1,padding = 1),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(channels[3]),
                torch.nn.MaxPool2d(kernel_size=2,stride=2)
                )

        self.block3_1 = torch.nn.Sequential(
                torch.nn.Conv2d(channels[3],channels[4],kernel_size = 3,stride = 1,padding = 1),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(channels[4]),
                )

        self.block3_2 = torch.nn.Sequential(
                torch.nn.Conv2d(channels[4],channels[5],kernel_size = 3,stride = 1,padding = 1),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(channels[5]),
                torch.nn.MaxPool2d(kernel_size=2,stride=2)
                )

        self.block4_1 = torch.nn.Sequential(
                torch.nn.Conv2d(channels[5],channels[6],kernel_size = 3,stride = 1,padding = 1),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(channels[6]),
                )

        self.block4_2 = torch.nn.Sequential(
                torch.nn.Conv2d(channels[6],channels[7],kernel_size = 3,stride = 1,padding = 1),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(channels[7])
                )
    def forward(self,input):
        out = self.block1_1(input)
        out = self.block1_2(out)
        out = self.block2_1(out)
        out = self.block2_2(out)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block4_1(out)
        out = self.block4_2(out)
        return out

# test
if __name__=="__main__":
    t = torch.randint(0,255,(1,1,240,320),dtype=torch.float32)
    
    config = {'backbone_type': 'VGG',
              'vgg': {'channels': [64, 64, 64, 64, 128, 128, 128, 128],
                      'convKernelSize': 3},
                      'det_head': {'feat_in_dim': 128},
                                   'des_head': {'feat_in_dim': 128,
                                                'feat_out_dim': 256}
              }

    net = VGGBackbone(config['vgg'],input_channel=1,device = 'mps')
    out = net(t)
    print(out.shape)
