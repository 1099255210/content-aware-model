from PIL import Image, ImageDraw, ImageFont
import os
import torch

from utils.read_data import MagazineData, layout_to_tensor

PALLATE = [
  '#6999eb',
  '#7ff8d1',
  '#3d9410',
  '#43f013',
  '#e6f11b',
  '#561a49',
  '#d4cf9e',
  '#b74672',
  '#b0708f',
  '#46261a',
  '#021939',
  '#dc1885',
  '#b9ed7d',
  '#18f9a0',
  '#0302cd',
  '#ce04dd',
  '#be4619',
  '#8e52bc',
  '#c53464',
  '#b1be36'
]

LAYER = {
  'text-over-image': 2,
  'headline-over-image': 2,
  'text': 1,
  'image': 1,
  'headline': 1,
  'background': 0,
}

PRESET = [
  {
    'dataset_name': 'magazine',
    'preset': {
      'text-over-image': '#6999eb',
      'headline-over-image': '#7ff8d1',
      'text': '#3d9410',
      'image': '#43f013',
      'headline': '#e6f11b',
      'background': '#561a49',
    }
  }
]

ORIGINAL_SIZE = [225, 300]
GRID_SIZE = [45, 60]
BLOCK_SIZE = [5, 5]

MAGAZINE_TAG = {
  'text': [0, 0, 1],
  'image': [0, 1, 0],
  'headline': [0, 1, 1],
  'text-over-image': [1, 0, 0],
  'headline-over-image': [1, 0, 1],
  'background': [1, 1, 0],
}


def recover_layout_from_label_name(label_name:str, screen_size:list[int, int]=[225, 300], root_dir='dataset', layout_folder='layout', dataset_name=''):

  '''
  Recover layout image from label_name in a specific dataset.
  '''

  img = Image.new('RGBA', screen_size, 'white')

  preset = {}
  for item in PRESET:
    if dataset_name == item['dataset_name']:
      preset = item['preset']


  magazine_dataset = MagazineData(root_dir, 'annotations', 'images')
  info = magazine_dataset.from_filename(label_name)
  layout = info['layout']
  layout_sorted = sorted(layout, key=lambda x: LAYER[x['label']])

  for item in layout_sorted:
    label = item['label']
    # bb = item['bb'].strip('[] ').split(',')
    # bb = [int(px) for px in bb]
    x1, y1, x2, y2 = item['bb']

    block = Image.new('RGBA', (x2 - x1, y2 - y1), preset[label])
    draw = ImageDraw.Draw(block)
    font = ImageFont.truetype(font="./fonts/wqy-microhei.ttc", size=8)
    draw.text((0, 0), label, font=font, fill=(0, 0, 0))

    img.paste(block, (x1, y1))
  img.save(os.path.join(root_dir, layout_folder, label_name + '.png'))
  print(os.path.join(root_dir, layout_folder, f'{label_name}.png'))


def recover_grid_layout_from_label_name(label_name:str, root_dir='dataset', layout_folder='layout', dataset_name=''):

  '''
  Recover grid layout image from label_name in a specific dataset.
  '''

  img = Image.new('RGBA', ORIGINAL_SIZE, 'white')

  preset = {}
  for item in PRESET:
    if dataset_name == item['dataset_name']:
      preset = item['preset']

  magazine_dataset = MagazineData(root_dir, 'annotations', 'images')
  info = magazine_dataset.from_filename(label_name)
  layout = info['layout']
  layout_sorted = sorted(layout, key=lambda x: LAYER[x['label']])

  for item in layout_sorted:
    label = item['label']
    x1, y1, x2, y2 = [round(px / 5) for px in item['bb']]
    
    block = Image.new('RGBA', ((x2 - x1) * 5, (y2 - y1) * 5), preset[label])
    draw = ImageDraw.Draw(block)
    font = ImageFont.truetype(font="./fonts/wqy-microhei.ttc", size=8)
    draw.text((0, 0), label, font=font, fill=(0, 0, 0))

    img.paste(block, (x1 * 5, y1 * 5))
  img.save(os.path.join(root_dir, layout_folder, label_name + '_grid.png'))
  print(os.path.join(root_dir, layout_folder, f'{label_name}_grid.png'))
  

def recover_grid_layout_from_tensor(tensor:torch.Tensor, root_dir='dataset', layout_folder='layout'):
  
  '''
  Recover grid layout image from tensor.
  '''

  img = Image.new('RGBA', ORIGINAL_SIZE, 'white')

  preset = PRESET[0]['preset']

  for y in range(0, tensor.shape[1]):
    for x in range(0, tensor.shape[2]):
      block_mask = []
      for idx in range(0, 3):
        block_mask.append(int(tensor[idx][y][x]))
      if block_mask == [0, 0, 0]:
        continue
      for key, val in MAGAZINE_TAG.items():
        if val == block_mask:
          color = preset[key]
      block = Image.new('RGBA', (5, 5), color)
      img.paste(block, (x * 5, y * 5))

  img.save(os.path.join(root_dir, layout_folder, 'recover_grid.png'))
  print(os.path.join(root_dir, layout_folder, 'recover_grid.png'))
  return


if __name__ == '__main__':
  name = 'fashion_0030'

  recover_layout_from_label_name(label_name=name, dataset_name='magazine')
  recover_grid_layout_from_label_name(label_name=name, dataset_name='magazine')


  root_dir = "dataset"
  magazine_dataset = MagazineData(root_dir, 'annotations', 'images')
  layout = magazine_dataset.from_filename(name)['layout']
  print(layout)
  tensor = layout_to_tensor(layout=layout)
  print(tensor)
  recover_grid_layout_from_tensor(tensor=tensor)