import yaml
import os

if __name__ == '__main__':
    steam = open('config/superpoint_COCO_train.yaml')
    config = yaml.safe_load(steam)
    print(config)