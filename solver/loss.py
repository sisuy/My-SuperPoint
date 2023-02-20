import torch
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F

def det_loss(keypoint_map,logits,grid_size,valid_mask,device):
    '''
    Parameters:
        keypoint_map: [B,H,W]
        logits: [B,C,H/gird_size,W/grid_size]
        valid_mask: [B,H,W]
        grid_size: int(default=8)
    '''
    keypoint_map = keypoint_map.unsqueeze(dim=1) # [1, 1, 240, 320]

    # pixel shuffle inverse 1 channels -> 64 channels
    pixelShuffle_inv = torch.nn.PixelUnshuffle(downscale_factor=grid_size)
    labels = pixelShuffle_inv(keypoint_map) # [1, 64, 30, 40]
    B,C,H,W = labels.shape

    # add dusbin(torch.ones -> [1,1,30,40])
    labels = torch.cat([2*labels,torch.ones([B,1,H,W],device=device)],dim=1) # [1, 65, 30, 40]
    labels = torch.argmax(labels + torch.zeros(labels.shape,device=device).uniform_(0,0.1),dim=1)#B*)H/grid_size)*(W/grid_size)

    # generate valid_mask: [B,H,W] -> [B,H/8,/W/8]
    if valid_mask is None:
        valid_mask = torch.ones([B,H,W])

    valid_mask = valid_mask.unsqueeze(dim=1) # [B,1,H,W] 
    valid_mask =  pixelShuffle_inv(valid_mask) # [B,64,H/8,W/8]
    valid_mask = torch.prod(valid_mask,dim=1).unsqueeze(dim=1) # [B,1,H/8,W/8]

    # use cross-entropy to get the loss
    lossFunction = torch.nn.CrossEntropyLoss(reduction='none')
    loss = lossFunction(logits,labels) # [1,30,40]
    valid_mask = valid_mask.squeeze(dim=1) # [1,30,40]

    # generate the loss covered by valid mask
    loss = torch.divide(torch.sum(valid_mask*loss,dim=(1,2)),
                        torch.sum(valid_mask + 1e-6,dim=(1,2)))
    return torch.mean(loss)

def warped_points(pixel_points,homography,device='cpu'):
    '''
    Parameters:
        pixel_points: [N,2]
        homography: [B,3,3]

    return points: [N,2]
    '''
    # homography batch processing
    if len(homography.shape) == 2:
        homography = homography.unsqueeze(dim=0) # [1,3,3]
    
    B = homography.shape[0]

    # Homogrous
    pixel_points = torch.cat([pixel_points,torch.ones([pixel_points.shape[0],1],device=device)],dim=1)
    pixel_points = torch.transpose(pixel_points,1,0) # [3,N]

    warped_points = torch.tensordot(homography,pixel_points,dims=[[2],[0]]) # [B,3,N]

    # normalisze: homogrous -> 2D
    warped_points = warped_points.transpose(1,2) # [B,N,3]
    warped_points = warped_points[:,:,:2]/warped_points[:,:,2:] # [B,1200,2]

    warped_points = torch.flip(warped_points,dims=(2,))
    warped_points = warped_points.squeeze(dim=0)
    return warped_points



def descriptor_loss(config,descriptor,warped_descriptor,homography,valid_mask=None,device='cpu'):
    """
    Parameter:
        descriptor: [B,C,H/8,W/8]
        warped_descriptor: [B,C,H/8,W/8]
        valid_mask: [B,H,W]
        homographic: [B,3,3]
    Return:
        loss: torch.float 
    """

    # transform the descriptor coordinate into warped_descriptor coordinate
    B,C,Hc,Wc = descriptor.shape
    grid_size = config['grid_size']
    lambda_d = config['loss']['lambda_d']
    lambda_loss = config['loss']['lambda_loss']
    positive_margin = config['loss']['positive_margin']
    negative_margin = config['loss']['negative_margin']

    pixel_coord = torch.stack(torch.meshgrid(torch.arange(Hc,device=device),torch.arange(Wc,device=device)),dim=-1) # [30,40,2]

    # compute the central pixel of the coord
    pixel_coord = pixel_coord*grid_size + grid_size//2
    pixel_points = torch.reshape(pixel_coord,(-1,2))

    warpedPixel_coord = warped_points(pixel_points,homography,device=device) # [N,2] if batch size==1, else [B,N,2]

    # reshape the coord tensor into the form like: [batch,Hc,Wc,1,1,2] and [batch,1,1,Hc,Wc,2]
    pixel_coord = torch.reshape(pixel_coord,[B,Hc,Wc,1,1,2])
    warpedPixel_coord = torch.reshape(warpedPixel_coord,[B,1,1,Hc,Wc,2])

    # TODO: Calculate the L2 norm
    cells_distance = torch.norm(warpedPixel_coord-pixel_coord,p='fro',dim=-1)
    
    # Calculate s
    s = (cells_distance-(grid_size-0.5)<=0).float() # [B,Hc,Wc,Hc,Wc]

    # descriptor reshape
    descriptor = torch.reshape(descriptor,[B,-1,Hc,Wc,1,1])
    warped_descriptor = torch.reshape(warped_descriptor,[B,-1,1,1,Hc,Wc])

    # descriptor normalization
    descriptor = F.normalize(descriptor,p=2,dim=1)
    warped_descriptor = F.normalize(warped_descriptor,p=2,dim=1)


    dot_product_descriptor = torch.sum(descriptor*warped_descriptor,dim=1)
    dot_product_descriptor = F.relu(dot_product_descriptor) # [B,Hc,Wc,Hc,Wc]

    # TODO: Maybe wrong here
    # use L2 norm to get average 1/(Hc*Wc)^2
    # TODO: why p=2? why not p = 1?
    dot_product_descriptor = torch.reshape(F.normalize(
                                            torch.reshape(dot_product_descriptor,[B,Hc,Wc,Hc*Wc]),
                                            p=2,
                                            dim=3),[B,Hc,Wc,Hc,Wc])
    dot_product_descriptor = torch.reshape(F.normalize(
                                            torch.reshape(dot_product_descriptor,[B,Hc*Wc,Hc,Wc]),
                                            p=2,
                                            dim=1),[B,Hc,Wc,Hc,Wc])
    positive_dist = torch.maximum(torch.tensor(0.,device=device),positive_margin-dot_product_descriptor)
    negative_dist = torch.maximum(torch.tensor(0.,device=device),dot_product_descriptor-negative_margin)
    loss = lambda_d*s*positive_dist + (1-s)*negative_dist # [B,Hc,Wc,Hc,Wc]

    # use mask to filter the keypoints
    # valid_mask: [B,Hc*grid_size,Wc*grid_size]
    valid_mask = torch.ones([B,Hc*grid_size,Wc*grid_sze],dtype=torch.float,device=device) if valid_mask is None else valid_mask

    # reshape it by using unshuffle_pixle
    valid_mask = torch.unsqueeze(valid_mask,dim=1).type(torch.float32) # [B,1,H,W]

    unshuffler = torch.nn.PixelUnshuffle(grid_size)
    valid_mask = unshuffler(valid_mask) # [B,C,Hc,Wc]
    valid_mask = torch.prod(valid_mask,dim=1) # [B,Hc,Wc]

    # copy debug from https://github.com/shaofengzeng/SuperPoint-Pytorch/blob/master/solver/loss.py
    normalization = torch.sum(valid_mask)*(Hc*Wc)

    positive_sum = torch.sum(valid_mask*lambda_d*s*positive_dist) / normalization
    negative_sum = torch.sum(valid_mask*(1-s)*negative_dist) / normalization

    print('positive_dist:{:.7f}, negative_dist:{:.7f}'.format(positive_sum, negative_sum))

    loss = lambda_loss*torch.sum(loss*valid_mask)/normalization
    print("descriptor loss: {}".format(loss))
    return loss


# test
if __name__=='__main__':
    device = 'cuda:0'
    # generate random keypoint map and pred result
    keypoint_map = torch.randint(-1,255,(1,240,320),dtype=torch.float,device=device)
    logits = torch.randint(-1,255,(1,65,30,40),dtype=torch.float,device=device)
    # generate valid mask
    valid_mask = torch.rand([1,240,320],dtype=torch.float32,device=device)
    # valid mask shape: [B,H,W]
    valid_mask = torch.where(valid_mask>-1.5,
                             torch.ones_like(valid_mask),
                             torch.zeros_like(valid_mask))

    loss = det_loss(keypoint_map,logits,8,valid_mask,device = 'cuda:0')
    print("Detector loss: {}".format(loss))

    # Test for descriptor loss	
    config = {  'grid_size': 8,
	 		    'loss': {'positive_margin': 1.0,
			  		    'negative_margin': 0.2,
					    'lambda_d': 0.05,
					    'lambda_loss': 10000},
				'epoch': 8,
				'base_lr': 0.001,
				'train_batch_size': 2,
				'test_batch_size': 2,
				'save_dir': './export/',
				'model_name': 'superpoint'}
    homography1 = torch.Tensor([[1,0,0],
                               [0.5,1,0],
                               [0.5,0,1]]).to(device)

    homography2 = torch.Tensor([[1,0,0],
                               [0.5,-2000,0],
                               [0.5,0,-1]]).to(device)
    
    homography = torch.stack([homography1,homography2],dim=0)
    print("batch homo: {}".format(homography.shape))

    # descriptor = torch.randint(-3,1,[1,65,30,40],dtype=torch.float,device=device)
    # warped_descriptor = torch.randint(-3,1,[1,65,30,40],dtype=torch.float,device=device)

    print('-----test1-----')
    descripto1 = torch.randint(-3,1,[1,65,30,40],dtype=torch.float,device=device)
    warped_descripto1 = torch.randint(-3,1,[1,65,30,40],dtype=torch.float,device=device)
    loss1 = descriptor_loss(config,descripto1,warped_descripto1,homography1,valid_mask=valid_mask,device='cuda:0')
    print("loss1: {}".format(loss1))
    print('-----test2-----')

    descriptor = torch.randint(-3,1,[1,65,30,40],dtype=torch.float,device=device)
    warped_descriptor = torch.randint(-3,1,[1,65,30,40],dtype=torch.float,device=device)
    loss2 = descriptor_loss(config,descriptor,warped_descriptor,homography2,valid_mask=valid_mask,device='cuda:0')
    print("loss2: {}".format(loss2))
    print('-----test3-----')

    descriptor = torch.randint(-3,1,[1,65,30,40],dtype=torch.float,device=device)
    warped_descriptor = torch.randint(-3,1,[1,65,30,40],dtype=torch.float,device=device)
    loss3 = descriptor_loss(config,descriptor,warped_descriptor,homography2,valid_mask=valid_mask,device='cuda:0')
    print("loss2: {}".format(loss3))
