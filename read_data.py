import os
import xml.etree.ElementTree as ET
import json
from torch.utils.data import Dataset

class GetData(Dataset):

  def __init__(self, root_dir, label_dir, image_dir):
    # Initialize
    self.root_dir = os.path.abspath(root_dir)
    self.label_dir = os.path.join(self.root_dir, label_dir)
    self.label_list = os.listdir(self.label_dir)
    self.image_dir = os.path.join(self.root_dir, image_dir)

  def __getitem__(self, idx):
    # Get Item
    label_item = self.label_list[idx]
    label_item_path = os.path.join(self.label_dir, label_item)
    ret = self.get_info(label_item_path)
    return ret


  def __len__(self):
    return len(self.label_list)
  

  def get_info(self, label_item_path):
    tree = ET.parse(label_item_path)
    root = tree.getroot()

    # Info
    filename = ''
    category = ''
    layoutsize = []
    layout = []
    keywords = []
    image_num = 0
    image_path_list = []

    # Get info
    for child in root:

      if child.tag == 'filename':
        filename = child.text

      if child.tag == 'category':
        category = child.text

      if child.tag == 'size':
        for item in child:
          layoutsize.append(int(item.text))

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
            layout.append({'label': label, 'bb': bb, 'area': area})
          except:
            ...

      if child.tag == 'text':
        for item in child:
          keywords.append(item.text)

    for n in range(image_num):
      image_path_list.append(
        os.path.join(self.image_dir, category, f'{filename}_{n + 1}.png')
      )

    info = {
      'filename': filename,
      'category': category,
      'layout': layout,
      'layout_size': layoutsize,
      'keywords': keywords,
      'image_num': image_num,
      'image_path_list': image_path_list,
    }

    return info


root_dir = "dataset"
magazine_dataset = GetData(root_dir, 'annotations', 'images')
for item in magazine_dataset:
  print(item['filename'])