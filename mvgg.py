import torch, torchvision
from torchvision.models import VGG16_Weights
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"

class imgNN(nn.Module):
  def __init__(self, originalModel):
    super(imgNN, self).__init__()
    self.features = nn.Sequential(*list(originalModel.features.children())[:-1])
    self.avgpool = nn.AdaptiveAvgPool2d(output_size = (1, 1))
    self.classifier = nn.Sequential(
      nn.Linear(512, 512),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(512, 256),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(256, 128)
    )
    
  def forward(self, img_list: List[str]):
    feat_list = []
    for p in img_list:
      t = process_img(p)
      feat_list.append(self.features(t))
    res = torch.stack(feat_list).sum(dim=0)
    res = self.avgpool(res)
    res = torch.flatten(res, 1)
    res = self.classifier(res)
    res = res.squeeze(0)
    return res
    
    
def image_encode(path: List[str], verbose=False) -> torch.Tensor:
  
  '''
  Images path list -> n * 512 * 14 * 14 -> 512 * 14 * 14 -> 512d -> 512d -> 256d -> 128d
  '''
  
  vgg16_pretrained = torchvision.models.vgg16(weights = VGG16_Weights.DEFAULT)
  model = imgNN(vgg16_pretrained).to(device)
  ret_vec = model(path)
  if verbose:
    print('---------------- Img2vec  result ----------------')
    torch.set_printoptions(threshold=10)
    print(ret_vec)
    print(f'Shape: {ret_vec.shape}')
    print('---------------- Img2vec  resend ----------------')
  return ret_vec


# class HookTool:
#   def __init__(self) -> None:
#     self.feat = None
    
#   def hook_feat(self, module, feat_in, feat_out):
#     self.feat = feat_out


# def get_feat_by_hook(model: nn.Module):
#   feat_hooks = []
#   for index, m in model.named_modules():
#     if index == "features.29":
#       cur_hook = HookTool()
#       m.register_forward_hook(cur_hook.hook_feat)
#       feat_hooks.append(cur_hook)
      
#   return feat_hooks


# def show_feature(feat_hooks):
#   print('The shape of feature is:', feat_hooks[0].feat.shape)
#   ft = feat_hooks[0].feat
#   ft = ft.squeeze(0)
#   gray_scale = torch.sum(ft, 0)
#   gray_scale = gray_scale / ft.shape[0]
#   plt.imshow(gray_scale.data.cpu().numpy())
#   plt.show()


def process_img(path:str) -> torch.Tensor:
  trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0., std=1.)
  ])
  img = Image.open(path).convert('RGB')
  img = trans(img).unsqueeze(0)
  return img.to(device)
  

if __name__ == '__main__':
  path_list = ['./assets/turtle.png', './assets/food.png']
  image_encode(path_list, verbose=True)
