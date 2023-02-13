import torch
import os

class SuperPointBNNet(torch.nn.Module):
    def __init__(self,config,input_chanel=1,grid_size = 8,device = 'cpu', using_bn = True):
        super(SuperPointBNNet,self).__init__()
        self.nms = config['nms']
        self.det_thresh = config['det_thresh']
        self.topk = config['topk']
