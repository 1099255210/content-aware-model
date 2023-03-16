'''
Some Settings, Consts.
'''

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

LAYER = {
  'text-over-image': 2,
  'headline-over-image': 2,
  'text': 1,
  'image': 1,
  'headline': 1,
  'background': 0,
}

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

PRESET = [
  {
    'dataset_name': 'magazine',
    'preset': {
      'text-over-image': '#6999eb',
      'headline-over-image': '#7ff8d1',
      'text': '#3d9410',
      'image': '#43f013',
      'headline': '#e6f11b',
      'background': '#ffffff',
    }
  }
]

