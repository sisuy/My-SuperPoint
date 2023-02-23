import torch
import torch
import cv2
from math import pi
import collections
import numpy as np
import torch.nn.functional as F

# solver function
def filter_points(pts,shape,device='cpu'):
    """
    params: pts [N,2]
    params: shape [H,W]
    return filter_points [N_filter,2]
    """
    if pts.shape[0] != 0:
        mask = (pts>=0) & (pts <= torch.tensor(shape,device=device)-1)
        mask = torch.all(mask,dim=1)
        return pts[mask]
    else:
        return pts

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

    pixel_points = torch.fliplr(pixel_points)
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

def dict_update(d, u):
    """Improved update for nested dictionaries.
    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.
    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def erosion2d(image,radius=0,border_value=1e6,device='cpu'):
    # create struring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2,)*2)
    kernel = torch.as_tensor(kernel[np.newaxis,:,:],device=device,dtype=torch.float) # [1,Hs,Ws]
    C,Hs,Ws = kernel.shape

    # flip
    kernel = torch.flip(kernel,dims=[1,2])
    
    # get the centre of the kernel
    origin = ((Hs-1)//2,(Ws-1)//2)

    # padding
    img_pad = F.pad(image,[origin[0],kernel.shape[1]-origin[0]-1,origin[1],kernel.shape[2]-origin[1]-1],mode='constant',value=border_value)

    # unfold the image
    img_unfold = F.unfold(img_pad,kernel_size=[Hs,Ws]) # [B,Hs*Ws,P]

    # strel [B,Hs*Ws] -> [B,Hs*Ws,1]
    strel_flatten = torch.flatten(kernel,start_dim=1) # [B,Hs*Ws]
    strel_flatten = strel_flatten.unsqueeze(dim=-1) # [B,Hs*Ws,1]


    # select the minimum value as the central pixel in a patch
    diff = img_unfold - strel_flatten # get the diffenreces between img_unfold and strel_element
    diff,_ = torch.min(diff,dim=1) # [1,Hs*Ws]
    
    # reshape: [1,Hs*Ws] -> [B,C,H,W]
    ret = torch.reshape(diff,image.shape)

    # fold the image
    return ret
