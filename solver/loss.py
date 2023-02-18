import torch

def det_loss(keypoint_map,logits,grid_size,valid_mask,device):
    '''
    Parameters:
        keypoint_map: [B,W,H]
        logits: [B,C,W/gird_size,H/grid_size]
        valid_mask: [B,W,H]
        grid_size: int(default=8)
    '''
    keypoint_map = keypoint_map.unsqueeze(dim=1) # [1, 1, 240, 320]

    # pixel shuffle inverse 1 channels -> 64 channels
    pixelShuffle_inv = torch.nn.PixelUnshuffle(downscale_factor=grid_size)
    labels = pixelShuffle_inv(keypoint_map) # [1, 64, 30, 40]
    B,C,W,H = labels.shape

    # add dusbin(torch.ones -> [1,1,30,40])
    labels = torch.cat([2*labels,torch.ones([B,1,W,H])],dim=1) # [1, 65, 30, 40]
    labels = torch.argmax(labels + torch.zeros(labels.shape,device=device).uniform_(0,0.1),dim=1)#B*)H/grid_size)*(W/grid_size)

    # generate valid_mask: [B,W,H] -> [B,W/8,/H/8]
    if valid_mask is None:
        valid_mask = torch.ones([B,W,H])

    valid_mask = valid_mask.unsqueeze(dim=1) # [B,1,W,H] 
    valid_mask =  pixelShuffle_inv(valid_mask) # [B,64,W/8,H/8]
    valid_mask = torch.prod(valid_mask,dim=1).unsqueeze(dim=1) # [B,1,W/8,H/8]

    # use cross-entropy to get the loss
    lossFunction = torch.nn.CrossEntropyLoss(reduction='none')
    loss = lossFunction(logits,labels) # [1,30,40]
    valid_mask = valid_mask.squeeze(dim=1) # [1,30,40]

    # generate the loss covered by valid mask
    loss = torch.divide(torch.sum(valid_mask*loss,dim=(1,2)),
                        torch.sum(valid_mask + 1e-6,dim=(1,2)))
    return torch.mean(loss)



# test
if __name__=='__main__':
    # generate random keypoint map and pred result
    keypoint_map = torch.randint(-1,255,(1,240,320),dtype=torch.float)
    logits = torch.randint(-1,255,(1,65,30,40),dtype=torch.float)
    # generate valid mask
    valid_mask = torch.rand([1,240,320],dtype=torch.float32)
    # valid mask shape: [B,W,H]
    valid_mask = torch.where(valid_mask>-1.5,
                             torch.ones_like(valid_mask),
                             torch.zeros_like(valid_mask))

    loss = det_loss(keypoint_map,logits,8,valid_mask,device = 'cpu')
    print("Detector loss: {}".format(loss))
    
