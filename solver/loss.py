import torch
import warnings
warnings.filterwarnings("ignore")

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

def warped_points(pixel_points,homography):
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
    pixel_points = torch.cat([pixel_points,torch.ones([pixel_points.shape[0],1])],dim=1)
    # print("Homogrous shape: {}".format(pixel_points.shape))
    pixel_points = torch.transpose(pixel_points,1,0) # [3,N]

    warped_points = torch.tensordot(homography,pixel_points,dims=[[2],[0]]) # [B,3,N]

    # normalisze: homogrous -> 2D
    warped_points = warped_points.transpose(1,2) # [B,N,3]
    # print("warped_points reshape: {}".format(warped_points.shape))
    warped_points = warped_points[:,:,:2]/warped_points[:,:,2:] # [B,1200,2]

    warped_points = torch.flip(warped_points,dims=(2,))
    warped_points = warped_points.squeeze(dim=0)
    return warped_points



def descriptor_loss(config,descriptor,warped_descriptor,homography,valid_mask=None,device='cpu'):
    """
    Parameter:
        descriptor: [B,C,W/8,H/8]
        warped_descriptor: [B,C,W/8,H/8]
        valid_mask: [B,W,H]
        homographic: [B,3,3]
    Return:
        loss: torch.float 
    """

    # transform the descriptor coordinate into warped_descriptor coordinate
    B,C,Wc,Hc = descriptor.shape
    grid_size = config['grid_size']

    pixel_coord = torch.stack(torch.meshgrid(torch.arange(Wc),torch.arange(Hc)),dim=-1) # [30,40,2]
    print("pixel_coord shape: {}".format(pixel_coord.shape))

    # compute the central pixel of the coord
    pixel_coord = pixel_coord*grid_size + grid_size//2
    pixel_points = torch.reshape(pixel_coord,(-1,2))
    print("pixel_points shape: {}".format(pixel_points.shape))

    warpedPixel_coord = warped_points(pixel_points,homography) # [N,2]
    print("warped_coord : {}".format(warpedPixel_coord.shape))




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
                               [0.5,0,1]])

    homography2 = torch.Tensor([[1,0,0],
                               [0.5,1,0],
                               [0.5,0,1]])
    
    homography = torch.stack([homography1,homography2],dim=0)
    print("batch homo: {}".format(homography.shape))


    descriptor = torch.rand([1,65,30,40],dtype=torch.float)
    warped_descriptor = torch.rand([1,65,30,40],dtype=torch.float)
    descriptor_loss(config,descriptor,warped_descriptor,homography1,valid_mask=valid_mask,device='cpu')

    
