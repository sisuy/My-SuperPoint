import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.SuperPoint import SuperPointBNNet
from dataset.hpatches import PatchesDataset 

def get_points_from_prob(prob_map):
    keypoints = np.where(prob_map>=0)

    # pts information: use np.stack keypoints x, keypoints y
    pts = np.stack([keypoints[0],
                    keypoints[1]],
                    axis=1)

    return pts

def keep_true_keypoints(points,H,shape):
    # project points by applying homography transform
    warped_points = warp_points(points,H)

    # transform [H,W] -> [x,y]
    warped_points = warped_points[:,[1,0]]

    # fileter points
    mask = (warped_points[:,0] > 0) & (warped_points[:,0] < shape[0]) & (warped_points[:,1] > 0) & (warped_points[:,1] < shape[1])

    return points[mask]

def warp_points(pts,H):
    # compute
    pts = np.concatenate((pts,np.ones([pts.shape[0],1])),axis=-1) # [N,3]
    pts = np.dot(pts,np.transpose(H))
    pts = pts[:,:2]/pts[:,2:]
    return pts

if __name__=="__main__":
    PATH = './config/detection_repeatability.yaml'
    with open(PATH,'r') as file:
        config = yaml.safe_load(file)

    data_dir = config['data']['export_dir']

    # load the numpy file
    data_list = os.listdir(data_dir) 
    det_thresh = 0.015
    
    # used to store repeatability
    repeatability = []

    for d in data_list:
        dataPath = os.path.join(data_dir,d)
        data = np.load(dataPath)
        # load data
        prob = data['prob']
        warped_prob = data['warp_prob']
        H = data['homography']

        # transform keypoint map into keypoint list
        keypoints = get_points_from_prob(prob) 
        warped_keypoints = get_points_from_prob(warped_prob)

        # keep true keypoints by using homography transform, detect whether the warped_points in the range of img shape 
        keypoints = keep_true_keypoints(keypoints,H,prob.shape)
        warped_keypoints = keep_true_keypoints(warped_keypoints,np.linalg.inv(H),warped_prob.shape)

        
