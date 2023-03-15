import torch
import torch.nn as nn
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
cate_tag = ['fashion', 'food', 'news', 'science', 'travel', 'wedding']
t_prop_seg = 7
i_prop_seg = 10


class AttrNN(nn.Module):
  def __init__(self):
    super(AttrNN, self).__init__()
    self.cate_nn = nn.Sequential(nn.Linear(60, 48))
    self.text_prop_nn = nn.Sequential(nn.Linear(70, 48))
    self.image_prop_nn = nn.Sequential(nn.Linear(100, 48))
    self.res_nn = nn.Sequential(nn.Linear(144, 32))
    
  def forward(self, cate_vec, t_prop_vec, i_prop_vec):
    cate_vec = self.cate_nn(cate_vec)
    t_prop_vec = self.text_prop_nn(t_prop_vec)
    i_prop_vec = self.image_prop_nn(i_prop_vec)
    res = torch.cat((cate_vec, t_prop_vec, i_prop_vec), dim=0)
    res = self.res_nn(res)
    return res
    
model = AttrNN().to(device)

def attr_encode(cate: str, t_prop: float, i_prop: float, verbose=False) -> torch.Tensor:
  
  '''
  Attributes -> 48d + 48d + 48d = 144d -> 32d
  '''
  
  cate_vec = cate2vec(cate)
  t_prop_vec = t_prop2vec(t_prop)
  i_prop_vec = i_prop2vec(i_prop)

  cate_vec = farray(cate_vec)
  t_prop_vec = farray(t_prop_vec)
  i_prop_vec = farray(i_prop_vec)

  ret_vec = model(cate_vec, t_prop_vec, i_prop_vec)
  if verbose:
    print('---------------- Attr2vec result ----------------')
    torch.set_printoptions(threshold=10)
    print(ret_vec)
    print(f'Shape: {ret_vec.shape}')
    print('---------------- Attr2vec resend ----------------')
  return ret_vec


def farray(array: np.ndarray) -> torch.Tensor:
  return torch.from_numpy(array).to(torch.float32).to(device)


def cate2vec(cate: str) -> np.ndarray:
  
  '''
  Category -> 60d
  '''
  
  ret_vec = np.zeros(shape=(len(cate_tag)))
  for idx, tag in enumerate(cate_tag):
    if tag == cate:
      ret_vec[idx] = 1
  ret_vec = np.tile(ret_vec, 10)
  return ret_vec


def t_prop2vec(prop: float) -> np.ndarray:
  
  '''
  Text Proportion -> 70d
  '''
  
  ret_vec = np.zeros(shape=(t_prop_seg))
  for idx in range(0, t_prop_seg):
    if idx == int(prop * 10):
      ret_vec[idx] = 1
  ret_vec = np.tile(ret_vec, 10)
  return ret_vec


def i_prop2vec(prop: float) -> np.ndarray:
    
  '''
  Image Proportion -> 70d
  '''
  
  ret_vec = np.zeros(shape=(i_prop_seg))
  for idx in range(0, i_prop_seg):
    if idx == int(prop * 10):
      ret_vec[idx] = 1
  ret_vec = np.tile(ret_vec, 10)
  return ret_vec


if __name__ == '__main__':
  cate = 'food'
  t_prop = 0.53
  i_prop = 0.11

  attr_encode(cate, t_prop, i_prop, verbose=True)