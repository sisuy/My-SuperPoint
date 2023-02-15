import torch

class DetectorHead(torch.nn.Module):
    def __init__(self,input_channel,grid_size,using_bn):
        super(DetectorHead,self).__init__()
        self.grid_size = grid_size
        self.using_bn = using_bn

        # build convolutional layers
