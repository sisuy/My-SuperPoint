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
    tensor = torch.reshape(tensor,(newNum,newChannel,height,scale_factor,width,scale_factor))
    tensor = tensor.permute(0,1,4,2,5,3)
    tensor = torch.reshape(tensor,(newNum,newChannel,newHeight,newWidth))
    return tensor



if __name__=='__main__':
    tensor = torch.randint(0,255,(1,64,30,40),dtype=torch.float32)
    out = pixel_shuffle(tensor,8)
    print(tensor.shape)
    print(out.shape)
