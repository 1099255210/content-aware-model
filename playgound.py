import torch 
import torch.nn as nn

# class Net(nn.Module):
#   def __init__(self):
#     super(Net, self).__init__()
#     self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),stride=(2,2),padding=(1,1))
#     self.bn1 = nn.BatchNorm1d(num_features=64)
      
  
#   def forward(self, x):
    
#     print(x.shape)
#     x = self.conv1(x)
#     print(x.shape)
#     x = self.bn1(x)
#     print(x.shape)
#     return x
  
# model = Net()
# x = torch.randn(3, 32, 32)
# out = model(x)


# real_layout_size = [225, 300]
# network_input_layout_size = [45, 60]
# one_block_size = [5, 5]

# layout = [
#   {
#     'label': 'text',
#     'bb': [151, 26, 205, 280],
#     'area': 13716
#   },
# ]

# x1, y1, x2, y2 = layout[0]['bb']  # x1=151, y1=26, x2=205, y2=280
# print(round(x1 / 5), round(y1 / 5), round(x2 / 5), round(y2 / 5))



# colors = [
#   '#6999eb',
#   '#7ff8d1',
#   '#3d9410',
#   '#43f013',
#   '#e6f11b',
#   '#561a49',
#   '#d4cf9e',
#   '#b74672',
#   '#b0708f',
#   '#46261a',
#   '#021939',
#   '#dc1885',
#   '#b9ed7d',
#   '#18f9a0',
#   '#0302cd',
#   '#ce04dd',
#   '#be4619',
#   '#8e52bc',
#   '#c53464',
#   '#b1be36'
# ]

def changed_block_data(data:torch.Tensor, bb:list[int, int], info:list[int, int]) -> torch.Tensor:

  '''
  Change 3-channel layout data.
  '''

  for idx, b in enumerate(info):
    for x in range(bb[0], bb[2] + 1):
      for y in range(bb[1], bb[3] + 1):
        try:
          data[idx][y][x] = b
        except:
          ...
  print(data)
  return data


x = torch.zeros((3, 10, 6), dtype=int)

changed_block_data(data=x, bb=[0, 0, 2, 4], info=[1, 1, 1])

  