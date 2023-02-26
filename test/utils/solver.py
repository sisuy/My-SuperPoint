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

def warped_points(points, homographies, device='cpu'):
    """
    :param points: (N,2), tensor
    :param homographies: [B, 3, 3], batch of homographies
    :return: warped points B,N,2
    """
    if len(points)==0:
        return points

    #TODO: Part1, the following code maybe not appropriate for your code
    points = torch.fliplr(points)
    if len(homographies.shape)==2:
        homographies = homographies.unsqueeze(0)
    B = homographies.shape[0]
    ##TODO: uncomment the following line to get same result as tf version
    # homographies = torch.linalg.inv(homographies)
    points = torch.cat((points, torch.ones((points.shape[0], 1),device=device)),dim=1)
    ##each row dot each column of points.transpose
    warped_points = torch.tensordot(homographies, points.transpose(1,0),dims=([2], [0]))#batch dot
    ##
    warped_points = warped_points.reshape([B, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    #TODO: Part2, the flip operation is combinated with Part1
    warped_points = torch.flip(warped_points,dims=(2,))
    #TODO: Note: one point case
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
