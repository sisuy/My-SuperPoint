## TODO
* Train process[Doing]
    * implement train process(train.py)
    * implement module[Finished]
        * VGG backbone(model/modules/VGGBackbone.py)[Finished]
        * Detector head(model/modules/CNNheads.py)[Finished]
            * should we implement a learned up-sampling method?
        * Descriptor head(model/modules/CNNheads.py)[Finished]
            * should we implement a learned up-sampling method?
* Dataset part
    * customized dataset(coco.py)[Doing]
        * data argumentation and homographt adapation



## Project architecture
```
├── README.md
├── config
│   └── superpoint_COCO_train.yaml
├── dataset
│   └── coco.py
├── export
│   └── superpoint.pth
├── model
│   ├── SuperPoint.py
│   ├── __pycache__
│   │   └── solver.cpython-39.pyc
│   ├── modules
│   │   ├── CNNheads.py
│   │   ├── VGGBackbone.py
│   │   └── __pycache__
│   │       ├── CNNheads.cpython-39.pyc
│   │       └── VGGBackbone.cpython-39.pyc
│   ├── solver.py
│   └── utils
│       ├── __pycache__
│       │   └── tensor_op.cpython-39.pyc
│       └── tensor_op.py
├── note
│   ├── assets
│   │   ├── Untitled 1.png
│   │   ├── Untitled 2.png
│   │   └── Untitled.png
│   └── note.md
├── requirements.txt
├── solver
└── train.py
```
