# Repvgg-packaged

Python packaged version of RepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)


## Install

```bash
$ pip install repvgg-pytorch
```

## RepVGG


## Usage for Training

```python

import torch
from repvgg_pytorch import get_RepVGG_func_by_name

repvgg_name="RepVGG-A0-backbone"
model_builder=get_RepVGG_func_by_name(repvgg_name)
model=model_builder(deploy=False)

x=torch.randn((1,3,512,512))
out=model(x) # (1, 512, 64, 64]

# save the training model weights for deployment
deploy_model = repvgg_model_convert(model, save_path='RepVGG_deploy.pth')

```

## Parameters

- `deploy`: bool.  
Use training or deployment mode



## Usage for Deployment

```python

import torch
from repvgg_pytorch import get_RepVGG_func_by_name

repvgg_name="RepVGG-A0-backbone"
model_builder=get_RepVGG_func_by_name(repvgg_name)
model=model_builder(deploy=True)
model.load_state_dict(torch.load('RepVGG_deploy.pth'))

x=torch.randn((1,3,512,512))
out=model(x) # (1, 512, 64, 64]

```

## RepVGG Plus


## Usage for Training

```python

import torch
from repvgg_pytorch.repvgg import get_RepVGG_func_by_name

repvgg_name="RepVGG-A0-backbone"
model_builder=get_RepVGG_func_by_name(repvgg_name)
model=model_builder(deploy=False)

x=torch.randn((1,3,512,512))
out=model(x) # (1, 512, 64, 64]

```

## Parameters

- `deploy`: bool.  
Use training or deployment mode



## Usage for Deployment

```python

import torch
from repvgg_pytorch.repvgg import get_RepVGG_func_by_name

repvgg_name="RepVGG-A0-backbone"
model_builder=get_RepVGG_func_by_name(repvgg_name)
model=model_builder(deploy=True)

x=torch.randn((1,3,512,512))
out=model(x) # (1, 512, 64, 64]

```

