import torch
import os
import torch.nn as nn
import yaml
from model.modules.VGGBackbone import VGGBackbone
from model.modules.CNNheads import DetectorHead,DescriptorHead
from model.solver import box_nms
# mainly copy from https://github.com/shaofengzeng/SuperPoint-Pytorch

class SuperPointBNNet(torch.nn.Module):
    def __init__(self,config,input_channel=1,grid_size = 8,device = 'cpu', using_bn = True):
        super(SuperPointBNNet,self).__init__()
        self.nms = config['nms']
        self.det_thresh = config['det_thresh']
        self.topk = config['topk']
        self.device = device

        # load backbone
        self.backbone = VGGBackbone(config['backbone']['vgg'],device=self.device)

        # load detectorHead and descriptorHead
        self.detectorHead = DetectorHead(config['backbone']['det_head']['feat_in_dim'],
                                           grid_size=grid_size,
                                           using_bn=using_bn,
                                           device=self.device)

        self.descriptorHead = DescriptorHead(config['backbone']['des_head']['feat_in_dim'],
                                             config['backbone']['des_head']['feat_out_dim'],
                                             grid_size=config['grid_size'],
                                             using_bn=config['using_bn'],
                                             device=self.device)
    def forward(self,input):
        # check input type
        if isinstance(input,dict):
            feat_map = self.backbone(input['img'])
        else:
            feat_map = self.backbone(input)

        # dection part
        det_outputs = self.detectorHead(feat_map)
        prob = det_outputs['prob']

        # TODO: use nms algorithm to filter the reduant keypoints
        if self.nms is not None:
            prob = [box_nms(p.unsqueeze(dim=0),
                            self.nms,
                            min_prob=self.det_thresh,
                            keep_top_k=self.topk).squeeze(dim=0) for p in prob]
            prob = torch.stack(prob)
            det_outputs.setdefault('prob_nms',prob)

        pred = prob[prob>=self.det_thresh]
        det_outputs.setdefault('pred',pred)


        # descriptor part
        desc_outputs = self.descriptorHead(feat_map)

        return {'det_info': det_outputs, 'desc_info': desc_outputs}




if __name__=='__main__':
    PATH = ''

    with open('./config/superpoint_COCO_train.yaml','r') as file:
        config = yaml.safe_load(file)
    net = SuperPointBNNet(config['model'],input_channel=1,grid_size = 8,device = 'mps', using_bn = True)
    tensor = torch.randint(0,255,[1,1,240,320],dtype=torch.float32)

    out = net(tensor) 
    print('intput shape: {}'.format(tensor.shape))
    print("detection output: {} \ndescriptor output: {}"\
            .format(out['det_info']['prob'].shape,out['desc_info']['desc'].shape))
    print('done')
