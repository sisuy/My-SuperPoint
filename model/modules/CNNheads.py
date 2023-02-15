import torch
from utils.tensor_op import pixel_shuffle

class DetectorHead(torch.nn.Module):
    def __init__(self,input_channel,grid_size,using_bn):
        super(DetectorHead,self).__init__()
        self.grid_size = grid_size
        self.using_bn = using_bn
        outputChannel = pow(self.grid_size,2)+1

        # build convolutional layers
        self.convPa = torch.nn.Conv2d(input_channel,256,3,stride=1,padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.BNPa = torch.nn.BatchNorm2d(256)

        # build (64+1)65 8*8 cells, 64 for smallest features in image, one used to be dusbin
        self.convPb = torch.nn.Conv2d(256,outputChannel,kernel_size=1,stride=1,padding=0)
        self.BNPb = torch.nn.BatchNorm2d(outputChannel)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self,input):
        # block1
        out = self.convPa(input)
        out = self.relu(out)
        self.BNPa(out)
    
        # block2
        out = self.convPb(out)
        out = self.relu(out)
        prob_map = self.BNPb(out)

        # apply softmax function
        prob_map = prob_map[:,:-1,:,:]
        prob_map = self.softmax(prob_map)
        prob_map = pixel_shuffle(prob_map,self.grid_size)

        # TODO: we need pixel shuffle to up-sample
        return prob_map
        


import yaml
if __name__=='__main__':
    # get random tensor which is in range of (0,255)
    ## (batch number,input channels,H,W)
    tensor = torch.randint(0,255,(1,128,30,40),dtype=torch.float32)

    det_net = DetectorHead(input_channel=128,grid_size=8,using_bn=True)

    out = det_net.forward(tensor)
    print(out.shape) # reshaped image size: [1,1,240,320]
