import yaml
import os
import torch
import argparse


if __name__ == '__main__':
    # solve enviroment problem
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.multiprocessing.set_start_method('spawn')

    # import train config file 
    parser = argparse.ArgumentParser()
    parser.add_argument('config')

    args = parser.parse_args()
    PATH = args.config

    with open(PATH,'r') as file:
        config = yaml.safe_load(file)

    
