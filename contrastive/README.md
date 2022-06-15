# CropMix + Asym-Siam

<p align="center">
  <img src="https://user-images.githubusercontent.com/2420753/161443048-ed1751ed-8a32-4d7d-85b7-024a6dc09067.png" width="300">
</p>

This repo is the code of CropMix on contrastive learning, we only slightly modify the [Asym-Siam repo](https://github.com/facebookresearch/asym-siam).

## Installation

1. [`Install git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 

2. Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

3. Install [apex](https://github.com/NVIDIA/apex) for the LARS optimizer used in linear classification. If you find it hard to install apex, it suffices to just copy the [code](https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py) directly for use.

4. Clone the repository: 
```
git clone https://github.com/JunlinHan/CropMix & cd CropMix/contrastive
```


## 1 Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

### MoCo + CropMix
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders] --enable-cropmix
```
By simply setting  **--enable-cropmix** to true, we can have asym CropMix on source side.


## 2 Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights, run:
```
python main_lincls.py \
  -a resnet50 \
  --lars \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [your checkpoint path] \
  [your imagenet-folder with train and val folders]
```

Our pre-trained models can be find at: https://drive.google.com/drive/folders/1x46eDfITqjUgnoogbkoqCepZtuNp60BM?usp=sharing

### License

The pre-training code is built on [MoCo](https://github.com/facebookresearch/moco), with additional designs described and analyzed in the paper.

The linear classification code is from [SimSiam](https://github.com/facebookresearch/simsiam), which uses LARS optimizer.

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### Asym-Siam paper
[Asym-Siam paper](https://arxiv.org/abs/2204.00613)

```
@inproceedings{wang2022asym,
  title     = {On the Importance of Asymmetry for Siamese Representation Learning},
  author    = {Xiao Wang and Haoqi Fan and Yuandong Tian and Daisuke Kihara and Xinlei Chen},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```
