import torch
import os
import torch.nn as nn
import yaml
from modules.VGGBackbone import VGGBackbone
from modules.CNNheads import DetectorHead

class SuperPointBNNet(torch.nn.Module):
    def __init__(self,config,input_channel=1,grid_size = 8,device = 'cpu', using_bn = True):
        super(SuperPointBNNet,self).__init__()
        self.nms = config['nms']
        self.det_thresh = config['det_thresh']
        self.topk = config['topk']

        # load backbone
        self.backbone = VGGBackbone(config['backbone']['vgg'])

        # load detectorHead and descriptorHead
        self.detectorHead(config['det_head']['feat_in_dim'],
                          grid_size=grid_size,
                          using_bn=using_bn)

if __name__=='__main__':
    PATH = ''

    with open('./config/superpoint_COCO_train.yaml','r') as file:
        config = yaml.safe_load(file)
    model = SuperPointBNNet(config['model'],input_channel=1,grid_size = 8,device = 'mps', using_bn = True)
    print('done')
