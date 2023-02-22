import torch 
import warnings
warnings.filterwarnings("ignore")
import os
from utils.photometric_augmentation import PhotoAugmentor
import glob
import cv2
import numpy as np
from utils.keypoint_op import compute_keypoint_map
from utils.homography_adaptation import homographic_aug_pipline
from copy import deepcopy

class COCODataset(torch.utils.data.Dataset):
    def __init__(self,config,is_train=False,device='cpu'):
        super(COCODataset,self).__init__()
        self.device = device
        self.is_train = is_train
        self.resize = tuple(config['resize'])
        self.photo_augmentor = PhotoAugmentor(config['augmentation']['photometric'])
        self.config = config

        # train or test
        if self.is_train == True:
            self.samples = self._init_data(config['image_train_path'],config['label_train_path'])
            print("COCO dataset size: {}".format(len(self.samples)))
        else:
            self.samples = self._init_data(config['image_test_path'],config['label_test_path'])
        self.__getitem__(0)

    # need to be implemented this function, nessearry for customized dataset
    ## load data into a list, use index to get the image and label of the sample image
    def _init_data(self,image_path,label_path=None):
        if not isinstance(image_path,list):
            image_paths,label_paths = [images_path],[label_path,]
        else:
            image_paths,label_paths =image_path,label_path

        image_types = ['jpg','jpeg','bmp','png']

        samples = []
        for im_path,lb_path in zip(image_paths,label_paths):
            for it in image_types:
                temp_im = glob.glob(os.path.join(im_path,'*.{}'.format(it))) # img.jpg/img.jpeg/img.bmp/img.png
                if lb_path is not None:
                    temp_lb = [os.path.join(lb_path,os.path.basename(imp)+'.npy') for imp in temp_im] 
                else:
                    temp_lb = [None,]*len(temp_im)
                temp = [{'image':imp,'label':lb} for imp,lb in zip(temp_im,temp_lb)]
                samples+=temp
        return samples

    ## return the size of the data length
    def __len__(self):
        return len(self.samples)

    ## return a image and a corresponsed label
    def __getitem__(self,idx):
        data_path = self.samples[idx]
        img = cv2.imread(data_path['image'],0)
        img = cv2.resize(img,self.resize[::-1])

        # load keypoints
        pts = None if data_path['label'] is None else np.load(data_path['label'])
        pts = pts.astype(np.float32) # [N,2]

        # transform numpy data into tensor
        img_tensor = torch.as_tensor(img.copy(),dtype=torch.float,device=self.device) # [H,W]
        H,W = img_tensor.shape
        pts = None if pts is None else torch.as_tensor(pts,dtype=torch.float,device=self.device) # [N,2]
        kpts_map = None if pts is None else compute_keypoint_map(pts,[H,W],device=self.device)
        valid_mask = torch.ones([H,W],device=self.device)

        data = {'raw':{'img':img_tensor,
                       'kpts':pts,
                       'kpts_map':kpts_map,
                       'mask':valid_mask},
                'warp':None,
                'homography':torch.eye(3,device=self.device),}
        data['warp'] = deepcopy(data['raw'])

        if self.is_train:
            photo_enable = self.config['augmentation']['photometric']['train_enable']
            homo_enable = self.config['augmentation']['homographic']['train_enable']
        else:
            photo_enable = self.config['augmentation']['photometric']['test_enable']
            homo_enable = self.config['augmentation']['homographic']['test_enable']

        # Homography adaptation
        if homo_enable == True and data['raw']['kpts'] is not None:
            data_homo = homographic_aug_pipline(data['warp']['img'],
                                                data['warp']['kpts'],
                                                self.config['augmentation']['homographic'],
                                                device=self.device)



# test dataset
if __name__=="__main__":
    is_train = True
    device = 'cuda:0'
    config = {'name': 'coco',
              'resize': [240, 320],
              'image_train_path': ['./data/images/train2017/'],
              'label_train_path': ['./data/labels/train2017/'],
              'image_test_path': ['./data/images/test2017/'],
              'label_test_path': ['./data/labels/test2017/'],
              'augmentation': {'photometric': {'train_enable': True, 'test_enable': True,
                                               'primitives': ['random_brightness',
                                                              'random_contrast',
                                                              'additive_speckle_noise',
                                                              'additive_gaussian_noise',
                                                              'additive_shade',
                                                              'motion_blur'],
                                               'params': {'random_brightness': {'max_abs_change': 50},
                                                          'random_contrast': {'strength_range': [0.5, 1.5]},
                                                          'additive_gaussian_noise': {'stddev_range': [0, 10]},
                                                          'additive_speckle_noise': {'prob_range': [0, 0.0035]},
                                                          'additive_shade': {'transparency_range': [-0.5, 0.5],
                                                                             'kernel_size_range': [100, 150],
                                                                             'nb_ellipses': 15},
                                                          'motion_blur': {'max_kernel_size': 3}}},
                               'homographic': {'train_enable': True,
                                               'test_enable': True,
                                               'params': {'translation': True,
                                                          'rotation': True,
                                                          'scaling': True,
                                                          'perspective': True,
                                                          'scaling_amplitude': 0.2, 
                                                          'perspective_amplitude_x': 0.2,
                                                          'perspective_amplitude_y': 0.2,
                                                          'patch_ratio': 0.85,
                                                          'max_angle': 1.5707963,
                                                          'allow_artifacts': True},
                                               'valid_border_margin': 3}}}
    dataset = COCODataset(config,is_train,device=device)
    # print("config: {}".format(config))
    
