import torch
from model.modules.utils.tensor_op import pixel_shuffle

class DetectorHead(torch.nn.Module):
    def __init__(self,input_channel,grid_size,using_bn,device='cpu'):
        super(DetectorHead,self).__init__()
        self.grid_size = grid_size
        self.using_bn = using_bn
        self.device = device
        outputChannel = pow(self.grid_size,2)+1

        # build convolutional layers
        self.convPa = torch.nn.Conv2d(input_channel,256,3,stride=1,padding=1,device=self.device)
        self.relu = torch.nn.ReLU(inplace=True)
        self.BNPa = torch.nn.BatchNorm2d(256,device=self.device)

        # build (64+1)65 8*8 cells, 64 for smallest features in image, one used to be dusbin
        self.convPb = torch.nn.Conv2d(256,outputChannel,kernel_size=1,stride=1,padding=0,device=self.device)
        self.BNPb = torch.nn.BatchNorm2d(outputChannel,device=self.device)

        self.softmax = torch.nn.Softmax(dim=1)

        # solver
        self.pixel_shuffle = torch.nn.PixelShuffle(upscale_factor=self.grid_size)

    def forward(self,input):
        # block1
        out = self.convPa(input)
        out = self.relu(out)
        out = self.BNPa(out)
    
        # block2
        out = self.convPb(out)
        out = self.BNPb(out)

        # apply softmax function
        prob_map = self.softmax(out)
        prob_map = prob_map[:,:-1,:,:] # [B,grid_size*grid_size,H/grid_size,W/grid_size]

        prob_map =self.pixel_shuffle(prob_map) # output size: [B,1,H,W]
        prob_map = prob_map.squeeze(dim=1) # [B,H,W]

        # we need pixel shuffle to up-sample
        return {'logit':out,'prob':prob_map}
        
class DescriptorHead(torch.nn.Module):
    def __init__(self,inputChannel,outputChannel,grid_size,using_bn=True,device='cpu'):
        super(DescriptorHead,self).__init__()
        self.inputChannel = inputChannel
        self.outputChannel = outputChannel
        self.grid_size = grid_size
        self.device = device

        # Convolutional layers
        D = 256
        self.convDa = torch.nn.Conv2d(inputChannel,D,kernel_size=3,stride=1,padding=1,device=self.device)
        self.convDb = torch.nn.Conv2d(D,256,kernel_size=1,stride=1,padding=0,device=self.device)

        # Activation function
        self.relu = torch.nn.ReLU(inplace=True)

        # Batch normalisation
        self.BNDa = torch.nn.BatchNorm2d(D,device=self.device)
        self.BNDb = torch.nn.BatchNorm2d(256,device=self.device)

        # up-sampler
        self.upSampler = torch.nn.Upsample(scale_factor=self.grid_size,mode='bicubic')

    def forward(self,input):
        # Block1
        out = self.convDa(input)
        out = self.relu(out)
        out = self.BNDa(out)

        # Block2
        out = self.convDb(out)
        out = self.BNDb(out) # [B,256,H/8,W/8]

        # Bicubic interpolation
        desc = self.upSampler(out)

        # L2-normalisation - Non-learned upsampling
        desc = torch.nn.functional.normalize(desc,p=2,dim=1) # why dim = 1?
        
        return {'desc_raw': out,'desc':desc}


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

    print('expected det output size: [1,240,320] \nexpected desc output size: [1,256,240,320]')

    print('detOut: {} \
           desOut: {}'.format(detOut['prob'].shape,desOut['desc'].shape))
    print('done')
