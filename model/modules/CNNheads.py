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

        prob_map = pixel_shuffle(prob_map,self.grid_size) # output size: [B,1,240,320]
        prob_map = prob_map.squeeze(dim=1)

        # TODO: we need pixel shuffle to up-sample
        return {'logit':out,'prob':prob_map}
        
class DescriptorHead(torch.nn.Module):
    def __init__(self,inputChannel,outputChannel,grid_size,using_bn=True):
        super(DescriptorHead,self).__init__()
        self.inputChannel = inputChannel
        self.outputChannel = outputChannel
        self.grid_size = grid_size

        # Convolutional layers
        D = 256
        self.convDa = torch.nn.Conv2d(inputChannel,D,kernel_size=3,stride=1,padding=1)
        self.convDb = torch.nn.Conv2d(D,256,kernel_size=1,stride=1,padding=0)

        # Activation function
        self.relu = torch.nn.ReLU(inplace=True)

        # Batch normalisation
        self.BNDa = torch.nn.BatchNorm2d(D)
        self.BNDb = torch.nn.BatchNorm2d(256)

    def forward(self,input):
        # Block1
        out = self.convDa(input)
        out = self.relu(out)
        out = self.BNDa(out)

        # Block2
        out = self.convDb(out)
        out = self.relu(out)
        desc_raw = self.BNDb(out)

        # TODO: Bitcubic interpolation

        # TODO: L2-normalisation - Non-learned upsampling
        
        return desc_raw


import yaml
if __name__=='__main__':
    config = {'name': 'superpoint',
              'using_bn': True,
              'grid_size': 8,
              'pretrained_model': 'none',
              'backbone': {'backbone_type': 'VGG',
                           'vgg': {'channels': [64, 64, 64, 64, 128, 128, 128, 128],
                                   'convKernelSize': 3},
                           'det_head': {'feat_in_dim': 128},
                           'des_head': {'feat_in_dim': 128,
                                        'feat_out_dim': 256}},
              'det_thresh': 0.001,
              'nms': 4,
              'topk': -1}
    # get random tensor which is in range of (0,255)
    ## (batch number,input channels,H,W)
    tensor1 = torch.randint(0,255,(1,128,30,40),dtype=torch.float32)
    tensor2 = torch.randint(0,255,(1,128,30,40),dtype=torch.float32)

    det_net = DetectorHead(input_channel=config['backbone']['det_head']['feat_in_dim'],\
                           grid_size=config['grid_size'],\
                           using_bn=config['using_bn'])
    des_net = DescriptorHead(inputChannel=config['backbone']['det_head']['feat_in_dim'],\
                             outputChannel=config['backbone']['des_head']['feat_out_dim'],\
                             grid_size=config['grid_size'],
                             using_bn='using_bn')

    detOut = det_net.forward(tensor1)
    desOut = des_net.forward(tensor2)

    print('detOut: {} \
           desOut: {}'.format(detOut['prob'].shape,desOut.shape))
