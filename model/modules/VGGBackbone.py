import torch
import numpy as np

class VGGBackbone(torch.nn.Module):
    def __init__(self,config,input_chanel = 1, device = 'cpu'):
        super(VGGBackbone,self).__init__()
        self.device = device
        channels = config['channels']

        self.block1_1 = torch.nn.Sequential(
                torch.nn.Conv2d(input_chanel,channels[0],kernel_size = 3,stride = 1,padding = 1),
                torch.nn.ReLU(inplace=True)
                )

        self.block1_2 = torch.nn.Sequential(
                torch.nn.Conv2d(channels[0],channels[1],kernel_size = 3,stride = 1,padding = 1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2)
                )

        self.block2_1 = torch.nn.Sequential(
                torch.nn.Conv2d(channels[1],channels[2],kernel_size = 3,stride = 1,padding = 1),
                torch.nn.ReLU(inplace=True)
                )

        self.block2_2 = torch.nn.Sequential(
                torch.nn.Conv2d(channels[2],channels[3],kernel_size = 3,stride = 1,padding = 1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2)
                )

        self.block3_1 = torch.nn.Sequential(
                torch.nn.Conv2d(channels[3],channels[4],kernel_size = 3,stride = 1,padding = 1),
                torch.nn.ReLU(inplace=True),
                )

        self.block3_2 = torch.nn.Sequential(
                torch.nn.Conv2d(channels[4],channels[5],kernel_size = 3,stride = 1,padding = 1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2)
                )

        self.block4_1 = torch.nn.Sequential(
                torch.nn.Conv2d(channels[3],channels[4],kernel_size = 3,stride = 1,padding = 1),
                torch.nn.ReLU(inplace=True),
                )

        self.block4_2 = torch.nn.Sequential(
                torch.nn.Conv2d(channels[4],channels[5],kernel_size = 3,stride = 1,padding = 1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2)
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
    config = {'channels': [64,64,64,64,128,128,128,128]}
    net = VGGBackbone(config,input_chanel = 1,device='mps')
    torch.save(net.state_dict(),'export/superpoint.pth')
    net.load_state_dict(torch.load('export/superpoint.pth'))

    for name,parameter in net.state_dict().items():
        print(name)
#        print(parameter)
    print('done')
