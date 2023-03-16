import os
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

'''
Self Created Imports
'''
from settings import MAGAZINE_TAG, LAYER


class MagazineData(Dataset):
  '''
  MagazineData Class.\n
  Get item returns a dict:\n
  'filename': filename,
  'category': category,
  'layout': layout,
  'layout_size': layout_size,
  'keywords': keywords,
  'image_num': image_num,
  'image_path_list': image_path_list,
  'text_area': text_area,
  'image_area': image_area,
  '''

  def __init__(self, root_dir, label_dir, image_dir):
    self.root_dir:str = os.path.abspath(root_dir)
    self.label_dir:str = os.path.join(self.root_dir, label_dir)
    self.label_list:list[str, str] = os.listdir(self.label_dir)
    self.image_dir:str = os.path.join(self.root_dir, image_dir)


  def __getitem__(self, idx):
    label_item = self.label_list[idx]
    label_item_path = os.path.join(self.label_dir, label_item)
    ret = self.get_info(label_item_path)
    return ret


  def __len__(self):
    return len(self.label_list)
  

  def get_info(self, label_item_path) -> dict:
    
    '''
    Given lable path, return all the info in a dict.\n
    If there're "Nan" or float numbers in the layout,
    return None.
    '''
    
    tree = ET.parse(label_item_path)
    root = tree.getroot()

    # Info
    filename = ''
    category = ''
    layout_size = []
    layout = []
    keywords = []
    image_num = 0
    image_path_list = []
    text_area = 0
    image_area = 0

    # Get info
    for child in root:

      if child.tag == 'filename':
        filename = child.text

      if child.tag == 'category':
        category = child.text

      if child.tag == 'size':
        for item in child:
          layout_size.append(int(item.text))

      if child.tag == 'layout':
        for element in child:
          label = element.get('label')
          if label == 'image':
            image_num += 1
          try:
            px = [int(i) for i in element.get('polygon_x').split(' ')]
            py = [int(i) for i in element.get('polygon_y').split(' ')]
            bb = [min(px), min(py), max(px), max(py)]
            area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            layout.append({
              'label': label,
              'bb': bb,
              'area': area
            })
          except:
            return None

      if child.tag == 'text':
        for item in child:
          keywords.append(item.text)

    for n in range(image_num):
      image_path = os.path.join(self.image_dir, category, f'{filename}_{n + 1}.png')
      if os.path.exists(image_path):
        image_path_list.append(image_path)
      else:
        image_num -= 1
      
    for item in layout:
      if item['label'] == 'text':
        text_area += int(item['area'])
      if item['label'] == 'image':
        image_area += int(item['area'])
      

    info = {
      'filename': filename,
      'category': category,
      'layout': layout,
      'layout_size': layout_size,
      'keywords': keywords,
      'image_num': image_num,
      'image_path_list': image_path_list,
      'text_area': text_area,
      'image_area': image_area,
    }

    return info
  

  def from_filename(self, filename:str):
    idx = self.label_list.index(f'{filename}.xml')
    label_item = self.label_list[idx]
    label_item_path = os.path.join(self.label_dir, label_item)
    ret = self.get_info(label_item_path)
    return ret


def read_pt(pt_folder='models', pt_name='multimodal.pt'):
  data = torch.load(os.path.join(pt_folder, pt_name))
  return data


def layout_to_tensor(layout:dict, tag:dict=MAGAZINE_TAG) -> torch.Tensor:

  ret_tensor = torch.zeros((3, 60, 45))
  layout_sorted = sorted(layout, key=lambda x: LAYER[x['label']])

  for item in layout_sorted:
    mask = MAGAZINE_TAG[item['label']]
    bb = [round(px / 5) for px in item['bb']]
    changed_block_data(tensor=ret_tensor, bb=bb, mask=mask)
    print(bb)
  
  return ret_tensor


def changed_block_data(tensor:torch.Tensor, bb:list[int, int], mask:list[int, int]) -> torch.Tensor:

  '''
  Change 3-channel layout data.
  '''

  for idx, b in enumerate(mask):
    for y in range(bb[1], bb[3] + 1):
      for x in range(bb[0], bb[2] + 1):
        try:
          tensor[idx][y][x] = b
        except:
          continue

  return tensor


if __name__ == '__main__':

  root_dir = "dataset"
  magazine_dataset = MagazineData(root_dir, 'annotations', 'images')
  layout = magazine_dataset.from_filename('fashion_0030')['layout']
  print(layout)
  tensor = layout_to_tensor(layout=layout)
  print(tensor)

  # stat = 0
  # for item in magazine_dataset:
  #   if item == None:
  #     continue
  #   if item['keywords'] == []:
  #     stat += 1
  #   print(item['filename'])
  # print(f'{stat}/{len(magazine_dataset)}')
  # print(magazine_dataset[20])
  # read_pt()