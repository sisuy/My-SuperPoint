import torch
import torch
import cv2
from math import pi
import collections
import numpy as np
import torch.nn.functional as F
from imgaug import augmenters as iaa

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
    strel = torch.flip(kernel,dims=[1,2])
    
    # get the centre of the kernel
    origin = ((Hs-1)//2,(Ws-1)//2)

    # padding
    image_pad = F.pad(image, [origin[0], strel.shape[1]-origin[0]-1, origin[1], strel.shape[2]-origin[1]-1], mode='constant', value=border_value)
    image_unfold = F.unfold(image_pad, kernel_size=strel.shape[1])#[B,C*sH*sW,L],L is the number of patches
    strel_flatten = torch.flatten(strel,start_dim=1).unsqueeze(-1)
    diff = image_unfold - strel_flatten
    # Take maximum over the neighborhood
    result, _ = diff.min(dim=1)
    # Reshape the image to recover initial shape
    return torch.reshape(result, image.shape)

def ratio_preserving_resize(img,target_size):
    scales = np.array((target_size[0]/img.shape[0], target_size[1]/img.shape[1]))##h_s,w_s

    new_size = np.round(np.array(img.shape)*np.max(scales)).astype(np.int)#
    temp_img = cv2.resize(img, tuple(new_size[::-1]))
    curr_h, curr_w = temp_img.shape
    target_h, target_w = target_size
    ##
    hp = (target_h-curr_h)//2
    wp = (target_w-curr_w)//2
    aug = iaa.Sequential([iaa.CropAndPad(px=(hp, wp, target_h-curr_h-hp, target_w-curr_w-wp),keep_size=False),])
    new_img = aug(images=temp_img)
    return new_img
