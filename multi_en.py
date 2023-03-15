from typing import List
import torch
import os

'''
Self Created Imports
'''
from utils.mattr import attr_encode
from utils.mvgg import image_encode
from utils.mw2v import w2v_encode
from utils.fusion import fusion
from utils.read_data import MagazineData


def multi_modal_embedding(
  image_path_list: List[str],
  keywords: List[str],
  cate: str,
  t_prop: float,
  i_prop: float,
  verbose = False
) -> torch.Tensor:
  
  '''
  image paths, keyword list, category, text prop, image prop -> 128d
  '''

  img_vec = image_encode(image_path_list, verbose=verbose)
  text_vec = w2v_encode(keywords, verbose=verbose)
  attr_vec = attr_encode(cate, t_prop, i_prop, verbose=verbose)

  res = fusion(img_vec, text_vec, attr_vec, verbose=verbose)
  
  print(res.shape)
  return res


def save_tensors(pt_folder = 'models', pt_name = 'multimodal.pt', root_dir = 'dataset', test = False, test_num = 30):

  '''
  Save multi-modal features into pt.
  '''

  if not os.path.exists(os.path.join(pt_folder, pt_name)):
    os.mkdir(os.path.join(pt_folder, pt_name))
  magazine_dataset = MagazineData(root_dir, 'annotations', 'images')

  tensors = {}
  count = 0
  for item in magazine_dataset:
    if item == None:
      continue
    
    count += 1
    if test and count == test_num:
      break
      
    filename = item['filename']
    image_path_list = item['image_path_list']
    keywords = item['keywords']
    category = item['category']
    full_area = item['layout_size'][0] * item['layout_size'][1]
    t_prop = item['text_area'] / full_area
    i_prop = item['image_area'] / full_area
     
    var = multi_modal_embedding(image_path_list, keywords, category, t_prop, i_prop, verbose=False).cpu().detach().numpy()
    tensors[filename] = torch.Tensor(var)
    print(item['filename'])
    
  torch.save(tensors, os.path.join(pt_folder, pt_name))


if __name__ == '__main__':
  
  # '''
  # Data Example
  # '''
  
  # image_path_list = ['./assets/turtle.png', './assets/food.png']
  # keywords = ['apple', 'technology', 'computer', 'simple']
  # category = 'food'
  # t_prop = 0.53
  # i_prop = 0.11
  
  # multi_modal_embedding(image_path_list, keywords, category, t_prop, i_prop, verbose=False)
  # multi_modal_embedding(image_path_list, keywords, category, t_prop, i_prop, verbose=False)
  # multi_modal_embedding(image_path_list, keywords, category, t_prop, i_prop, verbose=False)
  
  save_tensors()