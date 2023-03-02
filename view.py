import os
import yaml
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.SuperPoint import SuperPointBNNet
from dataset.hpatches import PatchesDataset 

def pickle_save(fname,data):
    with open(fname,'wb') as fout:
        pickle.dump(data,fout)

if __name__=="__main__":
    PATH = './config/descriptor_evaluation.yaml'
    with open(PATH,'r') as file:
        config = yaml.safe_load(file)
    device = 'cuda:0'

    out_dir = config['data']['export_dir']
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        print("remove directory {}".format(out_dir))
    os.makedirs(out_dir)

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
            pred1 = net(data['img'])
            pred2 = net(data['warped_img'])

            pred = {'prob': pred1['det_info']['prob_nms'],
                    'warp_prob': pred2['det_info']['prob_nms'],
                    'desc': pred1['desc_info']['desc'],
                    'warped_desc': pred2['desc_info']['desc'],
                    'homography': data['homography']}

            pred.update(data)

            # to numpy files
            pred = {k:v.detach().cpu().numpy().squeeze() for k,v in pred.items()}
            pred = {k: np.transpose(v,(1,2,0)) if k == 'warped_desc' or k == 'desc' else v for k,v in pred.items()} # desc or warped_desc shape: [H,W,C] 
            
            # export bin files
            filename = str(i)
            filepath = os.path.join(config['data']['export_dir'],'{}.bin'.format(filename))
            print("export file({}/{}): {}".format(i+1,len(dataloader),filepath))
            pickle_save(filepath,pred)
            time.sleep(1)
        print('done')
