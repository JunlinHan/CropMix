## Training
We provide our training code for image classification tasks (ImageNet recipe 1, CIFAR-10, and CIFAR-100). 


### CropMix configurations
```
scale:  min crop scale
mix_ratio: mixing ratio/weight for mixup
number: number of crops/cropping operations. support 2, 3, 4 and 234, where 234 is for (2,3,4)
operation: mixing operation, 0 for mixup, 1 for cutmix
inter_aug: apply intermediate augmentation (channel permutation)
```

### ImageNet
```
cd imagenet
python -W ignore main.py --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --scale 0.01 --mix_ratio 0.4 --number 234 --operation 0 --inter_aug\
[your imagenet-folder with train and val folders]
```

### CIFAR-100
```
cd cifar100
python -W ignore main.py --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --scale 0.5 --mix_ratio 0.4 --number 2 --operation 0 --inter_aug
```

### CIFAR-10
```
cd cifar10
python -W ignore main.py --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --scale 0.5 --mix_ratio 0.4 --number 2 --operation 0 --inter_aug
```

## Evaluation

For calibration, see [PixMix](https://github.com/andyzoujm/pixmix).

For adversarial attacks, see [Torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch). 

For corruption robustness, see [Co-Mixup](https://github.com/snu-mllab/Co-Mixup).

For ImageNet-A, see [Natural Adversarial Examples](https://github.com/hendrycks/natural-adv-examples).

For ImageNet-R, see [ImageNet-Rendition](https://github.com/hendrycks/imagenet-r).

For ImageNet-S, see [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch).

## Pre-trained models

We provide our pre-trained CropMix ImageNet classification models at: https://drive.google.com/drive/folders/1CmZW6UoR-YESqQ00IMPE9fVDfYjBa5IZ?usp=sharing


