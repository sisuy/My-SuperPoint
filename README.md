## TODO
* Train process
    * implement train process(train.py)
    * implement module
        * VGG backbone(model/modules/VGGBackbone.py)
        * Detector head(model/modules/CNN/CNNheads.py)
        * Descriptor head(model/modules/CNN/CNNheads.py)
* Dataset part
    * customized dataset(coco.py)


## data argumentation and homographt adapation

## Project architecture
├── README.md
├── config
│   └── superpoint_COCO_train.yaml
├── dataset
│   └── coco.py
├── model
│   ├── SuperPoint.py
│   └── modules
│       ├── CNNheads.py
│       └── VGGBackbone.py
├── solver
└── train.py
