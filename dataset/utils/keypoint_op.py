import torch
import numpy

def compute_keypoint_map(pts,shape,device='cpu'):
    """
    Parameters:
        pts: [N,2]
        shape: [H,W]
        device: str
    Return:
        keypoint_map: [H,W]
    """
    coord = torch.minimum(torch.round(pts).type(torch.int),torch.tensor(shape,device=device)-1) # limit the pixel not exceed the range of image
    kmap = torch.zeros((shape),dtype=torch.int, device=device)
    kmap[coord[:,0],coord[:,1]] = 1
    return kmap
