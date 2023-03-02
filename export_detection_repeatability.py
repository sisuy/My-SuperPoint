import os
import yaml
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.SuperPoint import SuperPointBNNet
from dataset.hpatches import PatchesDataset 

if __name__=="__main__":
    PATH = './config/detection_repeatability.yaml'
    with open(PATH,'r') as file:
        config = yaml.safe_load(file)
    device = 'cpu'

    if not os.path.exists(config['data']['export_dir']):
        print("Not exists {}".format(config['data']['export_dir']))
        os.makedirs(config['data']['export_dir'])
        print("Create export directory")

    # load dataset and generate dataloader
    dataset = PatchesDataset(config['data'],device=device)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=dataset.batch_collator)

    # load net
    net = SuperPointBNNet(config['model'],device=device)
    net.load_state_dict(torch.load(config['model']['pretrained_model'],map_location=device)) 
    net.to(device).eval()
    print("Network loading finished")

    with torch.no_grad():
        for i,data in tqdm(enumerate(dataloader)):
            prob1 = net(data['img'])
            prob2 = net(data['warped_img'])

            pred = {'prob': prob1['det_info']['prob_nms'],
                    'warp_prob': prob2['det_info']['prob_nms'],
                    'homography': data['homography']}

            pred.update(data)
            # to numpy files
            pred = {k:v.cpu().numpy().squeeze() for k,v in pred.items()}
            filename = str(i)
            filepath = os.path.join(config['data']['export_dir'],'{}.npz'.format(filename))
            print("export file({}/{}): {}".format(i+1,len(dataloader),filepath))
            np.savez_compressed(filepath,**pred)

    print("Done")
