import yaml
import os
import torch
import argparse
from model.SuperPoint import SuperPointBNNet
from torchviz import make_dot


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
    print(model)

