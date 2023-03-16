import torch
import torch.nn as nn
from torch import optim
import sys
sys.path.append('./utils')

'''
Self Created Imports
'''
from layout_gn import Encoder, Generator, Discriminator
from recover import recover_grid_layout_from_tensor
from utils.read_data import MagazineData, layout_to_tensor, read_pt


def test_gen():
  name = 'fashion_0030'

  root_dir = "dataset"
  magazine_dataset = MagazineData(root_dir, 'annotations', 'images')
  layout = magazine_dataset.from_filename(name)['layout']
  print(layout)
  tensor = layout_to_tensor(layout=layout)
  print(tensor)

  x = tensor
  data = read_pt()
  y = data[name]

  print(x.shape, y.shape)

  encoder = Encoder()
  generator = Generator()
  discriminator = Discriminator()

  z_hat = encoder(x, y)[0]

  x_tilde = generator(z_hat, y)
  x_tilde = x_tilde[:, 2:62, 9:54]
  print(x_tilde.shape, x_tilde)
  recover_grid_layout_from_tensor(tensor=x_tilde)


def train():
  ...


if __name__ == '__main__':
  test_gen()