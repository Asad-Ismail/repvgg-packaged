# Repvgg-packaged 🎁

A Python-packaged version of RepVGG: Making VGG-style ConvNets Great Again 🚀. The main contribution of this repo is to provide an easy-to-use backbone for RepVGG, which can be effortlessly used for downstream computer vision tasks

This project is based on the excellent original RepVGG implementation by authors Ding, Xiaohan et al. 🌟:
- [GitHub repository (code)](https://github.com/DingXiaoH/RepVGG)
- [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)


## Install

```bash
$ pip install repvgg-pytorch
```

## RepVGG


## Usage for Training

```python

import torch
from repvgg_pytorch import get_RepVGG_func_by_name
from repvgg_pytorch import repvgg_model_convert

repvgg_name="RepVGG-A0-backbone"
model_builder=get_RepVGG_func_by_name(repvgg_name)
model=model_builder(deploy=False)

x=torch.randn((1,3,512,512))
out=model(x) # (1, 512, 64, 64]

# Save the converted model for deployment
deploy_model = repvgg_model_convert(model, save_path='RepVGG_deploy.pth')

```



## Usage for Deployment

```python

import torch
from repvgg_pytorch import get_RepVGG_func_by_name

repvgg_name="RepVGG-A0-backbone"
model_builder=get_RepVGG_func_by_name(repvgg_name)
model=model_builder(deploy=True)
model.eval()
model.load_state_dict(torch.load('RepVGG_deploy.pth'))
x=torch.randn((1,3,512,512))
out=model(x) # (1, 512, 64, 64)

```

## RepVGG Plus


## Usage for Training

```python

import torch
from repvgg_pytorch.repvgg import get_RepVGG_func_by_name
from repvgg_pytorch import repvgg_model_convert

repvgg_name="RepVGG-A0-backbone"
model_builder=get_RepVGG_func_by_name(repvgg_name)
model=model_builder(deploy=False)

x=torch.randn((1,3,512,512))
out=model(x) # (1, 512, 64, 64]

# Save the converted model for deployment
deploy_model = repvgg_model_convert(model, save_path='RepVGGplus_deploy.pth')

```


## Usage for Deployment

```python

import torch
from repvgg_pytorch.repvgg import get_RepVGG_func_by_name

repvgg_name="RepVGG-A0-backbone"
model_builder=get_RepVGG_func_by_name(repvgg_name)
model=model_builder(deploy=True)
model.eval()

model.load_state_dict(torch.load('RepVGGplus_deploy.pth'))
x=torch.randn((1,3,512,512))
out=model(x) # (1, 320, 64, 64)

```

## References

```
@inproceedings{ding2021repvgg,
title={Repvgg: Making vgg-style convnets great again},
author={Ding, Xiaohan and Zhang, Xiangyu and Ma, Ningning and Han, Jungong and Ding, Guiguang and Sun, Jian},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
pages={13733--13742},
year={2021}
}
```