import torch
from repvgg_pytorch import get_RepVGG_func_by_name
from repvgg_pytorch import get_RepVGGplus_func_by_name
from repvgg_pytorch import repvgg_model_convert

# Input
x=torch.randn((1,3,512,512))

repvgg_name="RepVGG-A0-backbone"
model_builder=get_RepVGG_func_by_name(repvgg_name)
model=model_builder(deploy=False)
# First Out
out=model(x)
# You can use deploy model like this but most likely you will use it like the next one
deploy_model = repvgg_model_convert(model, save_path='RepVGG-A0-deploy.pth')

print(f"Output shape of RepVgG is {out.shape}")


repvgg_plus_name="RepVGGplus-backbone"
model_builder=get_RepVGGplus_func_by_name(repvgg_plus_name)
model=model_builder(deploy=False)
out=model(x)
deploy_model = repvgg_model_convert(model, save_path='RepVGGplus-deploy.pth')
print(f"Output shape of RepVgGplus is {out.shape}")


