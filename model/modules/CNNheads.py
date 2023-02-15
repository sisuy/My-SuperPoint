import torch

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
        prob_map = self.softmax(prob_map)
        return prob_map
        


import yaml
if __name__=='__main__':
    PATH = './config/superpoint_COCO_train.yaml'

    # with open(PATH,'r') as file:
    #     detConfig = yaml.safe_load(file)['model']['backbone']['det_head']
    #     desConfig = yaml.safe_load(file)['model']['backbone']['des_head']

    det_net = DetectorHead(1,8,True)
    tensor = torch.randint(0,255,(1,1,240,320),dtype=torch.float32)
    print(tensor)

    out = det_net.forward(tensor)
    print(out.shape)
