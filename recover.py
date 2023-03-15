from PIL import Image, ImageDraw
import os

from utils.read_data import MagazineData

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


def recover_layout_from_filename(filename:str, screen_size:list[int, int], root_dir='dataset', layout_folder='layout', dataset_name=''):
  img = Image.new('RGBA', screen_size, 'white')

  preset = {}
  for item in PRESET:
    if dataset_name == item['dataset_name']:
      preset = item['preset']


  magazine_dataset = MagazineData(root_dir, 'annotations', 'images')
  info = magazine_dataset.from_filename(filename)
  layout = info['layout']
  layout_sorted = sorted(layout, key=lambda x: LAYER[x['label']])
  print(layout_sorted)

  for item in layout:
    label = item['label']
    # bb = item['bb'].strip('[] ').split(',')
    # bb = [int(px) for px in bb]
    bb = item['bb']
    block = Image.new('RGBA', (bb[2] - bb[0], bb[3] - bb[1]), preset[label])
    # draw = ImageDraw.Draw(block)
    # font = ImageFont.truetype(font="./consola.ttf", size=20)
    # x = bb[0] + (bb[2] - bb[0]) / 2
    # y = bb[1] + (bb[3] - bb[1]) / 2

    # draw.text((0, 0), label, font=font, fill=(0, 0, 0))
    img.paste(block, (bb[0], bb[1]))
  img.save(os.path.join(root_dir, layout_folder, filename + '.png'))
  print(filename)


if __name__ == '__main__':
  recover_layout_from_filename(filename='fashion_0002', screen_size=[225, 300], dataset_name='magazine')