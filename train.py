import yaml
import os
import torch
import argparse
from torch.utils.data import DataLoader
from model.SuperPoint import SuperPointBNNet
from torchviz import make_dot
from dataset.coco import COCODataset
import warnings
from solver.loss import loss_fn
warnings.filterwarnings("ignore")

def train(config,model,dataloader,optimizier,device='cpu'):
    losses = []
    running_loss = 0.0

    for i, data in enumerate(dataloader['train'],0):
        # zero the parameter gradients
        optimizier.zero_grad()

        # forward
        raw_output = model(data['raw']['img'])
        warp_output = model(data['warp']['img'])

        prob,desc,prob_warp,desc_warp = raw_output['det_info'],\
                                        raw_output['desc_info'],\
                                        warp_output['det_info'],\
                                        warp_output['desc_info']
        # calculate loss
        loss = loss_fn(config,
                         prob,desc,
                         prob_warp,
                         desc_warp,
                         data,
                         device=device)





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
    dataloader = {"train": DataLoader(trainset,
                                      batch_size=config['solver']['train_batch_size'],
                                      shuffle=True,
                                      collate_fn=trainset.batch_collator),
                  "test": DataLoader(testset,
                                      batch_size=config['solver']['test_batch_size'],
                                      shuffle=True,
                                      collate_fn=testset.batch_collator)}

    # TODO: Load optimizier
    optimizier = torch.optim.Adam(model.parameters(),
                                  lr=config['solver']['base_lr'],
                                  betas=config['solver']['betas'])
    losses = train(config['solver'],model,dataloader,optimizier,device=device)
