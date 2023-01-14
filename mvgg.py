import torch, torchvision
from torchvision.models import VGG16_Weights
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

vgg16_pretrained = torchvision.models.vgg16(weights = VGG16_Weights.DEFAULT).cuda()
vgg16_pretrained.features = nn.Sequential(*list(vgg16_pretrained.features.children())[:-1]).cuda()
vgg16_pretrained.avgpool = nn.AdaptiveAvgPool2d(output_size = (1, 1)).cuda()
vgg16_pretrained.classifier = nn.Sequential(
  nn.Linear(512, 512, 1).cuda(),
  nn.ReLU(inplace=True).cuda(),
  nn.Dropout().cuda(),
  nn.Linear(512, 256, 1).cuda(),
  nn.ReLU(inplace=True).cuda(),
  nn.Dropout().cuda(),
  nn.Linear(256, 128, 1).cuda()
)

print(vgg16_pretrained)


class HookTool:
  def __init__(self) -> None:
    self.feat = None
    
  def hook_feat(self, module, feat_in, feat_out):
    self.feat = feat_out
    

def get_feat_by_hook(model: nn.Module):
  feat_hooks = []
  for index, m in model.named_modules():
    if index == "features.29":
      print(m)
      cur_hook = HookTool()
      m.register_forward_hook(cur_hook.hook_feat)
      feat_hooks.append(cur_hook)
      
  return feat_hooks


def process_img(path:str):
  trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0., std=1.)
  ])
  img = Image.open(path).convert('RGB')
  img = trans(img).unsqueeze(0)
  return img.to('cuda:0')
  
  
def show_feature(feat_hooks):
  print('The shape of feature is:', feat_hooks[0].feat.shape)
  ft = feat_hooks[0].feat
  ft = ft.squeeze(0)
  gray_scale = torch.sum(ft, 0)
  gray_scale = gray_scale / ft.shape[0]
  plt.imshow(gray_scale.data.cpu().numpy())
  plt.show()


if __name__ == '__main__':
  feat_hooks = get_feat_by_hook(vgg16_pretrained)
  x = process_img('./assets/turtle.png')
  out = vgg16_pretrained(x)
  show_feature(feat_hooks)
  print(out)
