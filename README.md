## Prepared

Put [weight.pth](https://drive.google.com/file/d/10I4V4Y1uW7YHNzTL_56fyCRhZbwlnSh6/view?usp=drive_link) into ./model file

## Installation
    conda create --name IM_web python=3.8
    conda activate --name IM_web
### Install Pytorch (Linux & windows)
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
### Install [MMsegmentation]([https://mmsegmentation.readthedocs.io/en/main/](https://mmsegmentation.readthedocs.io/en/main/get_started.html))
    pip install -U openmim
    mim install mmengine
    mim install mmcv==2.0.0
    cd mmsegmentation-main
    pip install -v -e .
### Install [MMdetection](https://mmdetection.readthedocs.io/en/latest/get_started.html)
    cd mmdetection
    pip install -v -e .

### Install Other Package
    pip install flask
    pip install timm
    pip install einops
    I forget others