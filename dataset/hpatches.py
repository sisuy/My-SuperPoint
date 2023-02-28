import os
import numpy as np
import torch
import cv2
from dataset.utils.solver import ratio_preserving_resize


class PatchesDataset(torch.utils.data.Dataset):
    def __init__(self,config,device='cpu'):
        super(PatchesDataset,self).__init__()
        self.config = config
        self.device = device

        """
        self.files:
            'image_paths': img_paths,
            'warped_image_paths': warpImg_paths,
            'homography_paths': homography_paths
        """
        self.files = self.__initDataset__()


    def __len__(self):
        return len(self.files['image_paths'])

    def __initDataset__(self):
        dataset_folder = self.config['data_dir']
        subfolders = [x for x in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder,x))]
        
        assert os.path.exists(dataset_folder)

        # generate path lists
        img_paths = []
        warpImg_paths = []
        homography_paths = []

        for sub_folder in subfolders:
            num_images = 7
            file_type = '.ppm'
            for i in range(2,num_images):
                img_paths.append(os.path.join(dataset_folder,sub_folder,"1"+file_type)) # 1.ppm
                warpImg_paths.append(os.path.join(dataset_folder,sub_folder,str(i)+file_type)) # (2-6).ppm
                homography_paths.append(os.path.join(dataset_folder,sub_folder,'H_1_'+str(i)))

        files = {'image_paths': img_paths,
                 'warped_image_paths': warpImg_paths,
                 'homography_paths': homography_paths}
        return files

    def __getitem__(self,idx):
        """
        self.files:
            'image_paths': img_paths,
            'warped_image_paths': warpImg_paths,
            'homography_paths': homography_paths
        """
        # get data from path lists
        img_path = self.files['image_paths'][idx]
        warped_img_path = self.files['warped_image_paths'][idx]
        homography_path = self.files['homography_paths'][idx]

        # read image by cv2
        img = cv2.imread(img_path,0)
        warped_img = cv2.imread(warped_img_path,0)
        homography = np.loadtxt(homography_path)

        # resize
        if self.config['preprocessing']['resize']:
            img_shape = img.shape
            warped_img_shape = warped_img.shape
            homography = {'homography': homography,
                          'img_shape': np.array(img_shape),
                          'warped_img_shape': np.array(warped_img_shape)}
            
            homography = self.adapt_homography_to_preprocessing(homography)
        img = self._preprocessing(img)
        warped_img = self._preprocessing(warped_img)

        # to torch tensor
        img = torch.as_tensor(img,dtype=torch.float32,device=self.device)
        warped_img = torch.as_tensor(warped_img,dtype=torch.float32,device=self.device)
        homography = torch.as_tensor(homography,dtype=torch.float32,device=self.device)

        # normalisze
        img = img/255
        warped_img = warped_img/255

        data = {"img":img,
                "warped_img": warped_img,
                "homography": homography}
        return data

    def _preprocessing(self,img):
        img = ratio_preserving_resize(img,self.config['preprocessing']['resize'])
        return img

    def adapt_homography_to_preprocessing(self,zip_data):
        """
        zip_data:
            'homography': homography,
            'img_shape': np.array(img_shape),
            'warped_img_shape': np.array(warped_img_shape)}
        """
        H = zip_data['homography'].astype(np.float32)
        source_size = zip_data['img_shape'].astype(np.float32)
        source_warped_size = zip_data['warped_img_shape'].astype(np.float32)
        target_size = self.config['preprocessing']['resize']
        
        s = np.max(target_size/source_size)
        up_scale = np.diag([1./s,1./s,1])

        warped_s = np.max(target_size/source_warped_size)
        down_scale = np.diag([warped_s,warped_s,1])

        pad_y,pad_x = (source_size*s - target_size)//2.0
        translation = np.array([[1,0,pad_x],
                                [0,1,pad_y],
                                [0,0,1]],dtype=np.float32)

        pad_y,pad_x = (source_warped_size*warped_s - target_size)//2.0
        warped_translation = np.array([[1,0,-pad_x],
                                       [0,1,-pad_y],
                                       [0,0,1]],dtype=np.float32)
        H = warped_translation @ down_scale @ H @ up_scale @ translation
        return H
    
    def batch_collator(self,samples):
        batch = {'img': [],
                 'warped_img': [],
                 'homography': []}
        for s in samples:
            for k,v in s.items():
                if 'img' in k:
                    batch[k].append(v.unsqueeze(dim=0))
                else:
                    batch[k].append(v)
        
        for i in batch:
            batch[i] = torch.stack(batch[i],dim=0)

        return batch


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    import yaml
    PATH = './config/detection_repeatability.yaml'
    with open(PATH,'r') as file:
        config = yaml.safe_load(file)
    device = 'cuda:0'
    dataset = PatchesDataset(config['data'],device=device)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=dataset.batch_collator)

    for i,data in enumerate(dataloader):
        if i >= 3:
            img = (data['img']*255).squeeze().cpu().numpy().astype(np.int).astype(np.uint8)
            warped_img = (data['warped_img']*255).squeeze().cpu().numpy().astype(np.int).astype(np.uint8)

            img = cv2.merge([img,img,img])
            warped_img = cv2.merge([warped_img,warped_img,warped_img])
            
            plt.subplot(1,2,1)
            plt.imshow(img)

            plt.subplot(1,2,2)
            plt.imshow(warped_img)

            plt.show()

    print('done')
