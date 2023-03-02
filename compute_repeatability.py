import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.SuperPoint import SuperPointBNNet
from dataset.hpatches import PatchesDataset 

def get_points_from_prob(prob,det_thresh):
    keypoints = np.where(prob>=det_thresh)
    print(keypoints)

def compute_repeatability(data,keep_k_points=3000,det_thresh=3):
        """
        data: 
            'img'
            'warped_img':
            'homography':
            'prob':
            'warp_prob':
        """


if __name__=="__main__":
    PATH = './config/detection_repeatability.yaml'
    with open(PATH,'r') as file:
        config = yaml.safe_load(file)
    device = 'cpu'

    data_dir = config['data']['export_dir']

    # load the numpy file
    data_list = os.listdir(data_dir) 
    det_thresh = 0.015
    for i in range(5):
        dataPath = os.path.join(data_dir,"{}.npz".format(i))
        data = np.load(dataPath)
        # load data
        img = data['img']
        warped_img = data['warped_img']
        H = data['homography']

        # prob
        prob = data['prob']
        warped_prob = data['warp_prob']

        # transform keypoint map into keypoint list
        keypoints = get_points_from_prob(prob,det_thresh) 

