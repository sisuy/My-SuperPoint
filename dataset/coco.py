import torch


class COCODataset(torch.utils.data.Dataset):
    def __init__(self):
        super(COCODataset,self).__init__()
        self.sample = None
    # need to be implemented this function, nessearry for customized dataset

    ## load data into a list, use index to get the image and label of the sample image
    def _init_data(self):

    ## return the size of the data length
    def __len__(self):

    ## return a image and a corresponsed label
    def __getitem__(self):
