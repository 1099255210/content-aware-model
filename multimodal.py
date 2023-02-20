from typing import List

'''
Self Created Functions
'''
from mattr import attr_encode
from mvgg import image_encode
from mw2v import w2v_encode
from fusion import fusion


def multiModalEmbedding(
  path_list: List[str],
  word_list: List[str],
  cate: str,
  t_ratio: float,
  i_ratio: float,
  verbose = False
):
  
  '''
  image paths, keyword list, category, text ratio, image ratio -> 128d
  '''

  img_vec = image_encode(path_list, verbose=verbose)
  text_vec = w2v_encode(word_list, verbose=verbose)
  attr_vec = attr_encode(cate, t_ratio, i_ratio, verbose=verbose)

  res = fusion(img_vec, text_vec, attr_vec, verbose=verbose)
  return res


if __name__ == '__main__':
  
  # Example
  path_list = ['./assets/turtle.png', './assets/food.png']
  word_list = ['apple', 'technology', 'computer', 'simple']
  cate = 'food'
  t_ratio = 0.53
  i_ratio = 0.11
  
  multiModalEmbedding(path_list, word_list, cate, t_ratio, i_ratio, verbose=True)