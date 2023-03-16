from PIL import Image, ImageDraw, ImageFont
import os
import torch

'''
Self Created Imports
'''
from utils.read_data import MagazineData, layout_to_tensor
from utils.settings import LAYER, PRESET, ORIGINAL_SIZE, BLOCK_SIZE, MAGAZINE_TAG


def recover_layout_from_label_name(
  label_name:str,
  screen_size:list[int, int]=[225, 300],
  root_dir='dataset',
  layout_folder='layout',
  dataset_name=''
):

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
  

def recover_grid_layout_from_tensor(tensor:torch.Tensor, root_dir='dataset', layout_folder='layout', save_name='recover_grid.png'):
  
  '''
  Recover grid layout image from tensor.
  '''

  img = Image.new('RGBA', ORIGINAL_SIZE, 'white')

  preset = PRESET[0]['preset']
  px, py = BLOCK_SIZE

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
      block = Image.new('RGBA', (px, py), color)
      img.paste(block, (x * px, y * py))

  img.save(os.path.join(root_dir, layout_folder, save_name))
  print(os.path.join(root_dir, layout_folder, save_name))
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