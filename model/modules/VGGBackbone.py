import torch

class VGGBackbone(torch.nn.Module):
    def __init__(self,config,input_chanel = 1, device = 'cpu'):
        super(VGGBackbone,self).__init__()
        self.device = device
        channels = config['channles']
