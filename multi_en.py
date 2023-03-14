from typing import List
import torch

'''
Self Created Imports
'''
from utils.mattr import attr_encode
from utils.mvgg import image_encode
from utils.mw2v import w2v_encode
from utils.fusion import fusion


def multiModalEmbedding(
  path_list: List[str],
  word_list: List[str],
  cate: str,
  t_ratio: float,
  i_ratio: float,
  verbose = False
) -> torch.Tensor:
  
  '''
  image paths, keyword list, category, text ratio, image ratio -> 128d
  '''

  img_vec = image_encode(path_list, verbose=verbose)
  text_vec = w2v_encode(word_list, verbose=verbose)
  attr_vec = attr_encode(cate, t_ratio, i_ratio, verbose=verbose)

  res = fusion(img_vec, text_vec, attr_vec, verbose=verbose)
  
  print(res.shape)
  return res


if __name__ == '__main__':
  
  # Data Example
  path_list = ['./assets/turtle.png', './assets/food.png']
  word_list = ['apple', 'technology', 'computer', 'simple']
  cate = 'food'
  t_ratio = 0.53
  i_ratio = 0.11
  
  multiModalEmbedding(path_list, word_list, cate, t_ratio, i_ratio, verbose=False)
  multiModalEmbedding(path_list, word_list, cate, t_ratio, i_ratio, verbose=False)
  multiModalEmbedding(path_list, word_list, cate, t_ratio, i_ratio, verbose=False)