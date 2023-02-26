import torch 
import numpy


# TODO: implement pixel_shuffle
def pixel_shuffle(tensor, scale_factor):
    num,channel,height,width = tensor.shape

    # check if the scale_factor is valid
    assert(channel%(scale_factor*scale_factor)==0)

    # update parameters 
    newNum = num
    newChannel = channel//(scale_factor*scale_factor)
    newHeight = height*scale_factor
    newWidth = width*scale_factor

    # shuffule
    tensor = torch.reshape(tensor,(newNum,newChannel,scale_factor,scale_factor,height,width))
    tensor = tensor.permute(0,1,4,2,5,3)
    tensor = torch.reshape(tensor,(newNum,newChannel,newHeight,newWidth))
    return tensor

def pixel_shuffle_inv(tensor,scale_factor):
    """
    Parameters: Tensor shape [N,C,W,H], int scale_factor
    Returns: Tensor shape [N,C*scale_factor*scale_factor,W/scale_factor,H/scale_factor]
    """
    num,channel,height,width = tensor.shape

    # check if the scale_factor is valid
    assert(width%scale_factor==0 and height%scale_factor==0)

    # update parameters 
    newNum = num
    newChannel = channel*pow(scale_factor,2)
    newHeight = height//scale_factor
    newWidth = width//scale_factor

    # shuffule
    tensor = torch.reshape(tensor,(newNum,channel,newHeight,scale_factor,newWidth,scale_factor))
    tensor = tensor.permute(0,1,3,5,2,4)
    tensor = torch.reshape(tensor,(newNum,newChannel,newHeight,newWidth))
    return tensor

if __name__=='__main__':
    tensor = torch.randint(0,255,(1,64,30,40),dtype=torch.float32)
    out1 = pixel_shuffle(tensor,8)
    out2 = pixel_shuffle_inv(out1,8)
    print('pixel_shuffle: {} \npixel_shuffle_inv: {}'.format(out1.shape,out2.shape))
