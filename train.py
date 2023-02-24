import yaml
import os
import torch
import argparse
from torch.utils.data import DataLoader
from model.SuperPoint import SuperPointBNNet
from torchviz import make_dot
from dataset.coco import COCODataset
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # solve enviroment problem
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.multiprocessing.set_start_method('spawn')

    # import train config file 
    parser = argparse.ArgumentParser()
    parser.add_argument('config')

    args = parser.parse_args()
    PATH = args.config
    assert os.path.exists(PATH)

    with open(PATH,'r') as file:
        config = yaml.safe_load(file)

    # If not exists the export directory, then creat it 
    if os.path.exists(config['solver']['save_dir']) == False:
        os.mkdir(config['solver']['save_dir'])

    # Load superpoint net
    device = 'mps'
    x = torch.randint(0,255,[1,1,240,320],dtype=torch.float,device=device)
    model = SuperPointBNNet(config['model'],device=device)

    # TODO: Load dataset
    trainset = COCODataset(config['data'],is_train=True,device=device)
    testset = COCODataset(config['data'],is_train=False,device=device)

    # build dataloader
    train_batchSize = 1
    test_batchSize = 1
    dataloader = {"train": DataLoader(trainset,
                                      batch_size=train_batchSize,
                                      shuffle=True,
                                      collate_fn=trainset.batch_collator),
                  "test": DataLoader(testset,
                                      batch_size=test_batchSize,
                                      shuffle=True,
                                      collate_fn=testset.batch_collator)}

    # TODO: Load optimizier
