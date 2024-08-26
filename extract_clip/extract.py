# import clip.clip as clip
import clip
import torch
from collections import OrderedDict

model, _ = clip.load("ViT-B/16", device='cuda')
new_state_dict = OrderedDict()
for k, v in model.state_dict().items():
    if 'visual.' in k:
        if k[7:] not in ["proj", "ln_post.weight", "ln_post.bias"]:
            new_state_dict[k[7:]] = v

torch.save(new_state_dict, 'vit_b16.pth')

model, _ = clip.load("ViT-L/14", device='cuda')
new_state_dict = OrderedDict()
for k, v in model.state_dict().items():
    if 'visual.' in k:
        if k[7:] not in ["proj", "ln_post.weight", "ln_post.bias"]:
            new_state_dict[k[7:]] = v

torch.save(new_state_dict, 'vit_l14.pth')

model, _ = clip.load("ViT-L/14@336px", device='cuda')
new_state_dict = OrderedDict()
for k, v in model.state_dict().items():
    if 'visual.' in k:
        if k[7:] not in ["proj", "ln_post.weight", "ln_post.bias"]:
            new_state_dict[k[7:]] = v

torch.save(new_state_dict, 'vit_l14_336.pth')