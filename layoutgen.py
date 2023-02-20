import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class LGN(nn.Module):
  def __init__(self):
    super(LGN, self).__init__()
    
    # Layout Encoder
    self.pad_1 = nn.ZeroPad2d(padding=(2, 2, 9, 10, 0, 0))
    self.pad_2 = nn.ZeroPad2d(padding=(2, 2, 9, 10, 0, 0))
    self.c_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    self.c_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    self.bn_2 = nn.BatchNorm1d(16)
    self.c_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    self.bn_3 = nn.BatchNorm1d(8)
    self.c_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    self.bn_4 = nn.BatchNorm1d(4)
    self.c_5 = nn.Conv2d(in_channels=640, out_channels=128, kernel_size=(4, 4), stride=(1, 1))
    self.c_6 = nn.Conv2d(in_channels=640, out_channels=128, kernel_size=(4, 4), stride=(1, 1))
    self.flatten = nn.Flatten()
    self.leakyrelu = nn.LeakyReLU()
    
    # Layout Generator
    self.fc_1 = nn.Linear(in_features=256, out_features=8192)
    self.bn_5 = nn.BatchNorm1d(4)
    self.dc_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(5, 5), stride=(2, 2))
    self.bn_6 = nn.BatchNorm1d(8)
    self.dc_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(5, 5), stride=(2, 2))
    self.bn_7 = nn.BatchNorm1d(16)
    self.dc_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=(2, 2))
    self.bn_8 = nn.BatchNorm1d(32)
    self.dc_4 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(5, 5), stride=(2, 2))
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()
    
    # Layout Discriminator
    self.c_7 = nn.Conv2d(in_channels=131, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    self.c_8 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    self.bn_9 = nn.BatchNorm1d(16)
    self.c_9 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    self.bn_10 = nn.BatchNorm1d(8)
    self.c_10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    self.bn_11 = nn.BatchNorm1d(4)
    self.c_11 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(4, 4), stride=(1, 1))
    self.fc_2 = nn.Linear(in_features=256, out_features=1)
    
    
  def encoder(self, x, y):
    '''
    In: 64*64*3, out: 128, 128
    '''
    x = self.pad_1(x)
    x = self.c_1(x)
    x = self.leakyrelu(x)
    x = self.c_2(x)
    x = self.bn_2(x)
    x = self.leakyrelu(x)
    x = self.c_3(x)
    x = self.bn_3(x)
    x = self.leakyrelu(x)
    x = self.c_4(x)
    x = self.bn_4(x)
    x = self.leakyrelu(x)
    y = y.unsqueeze(1).unsqueeze(1)
    y = y.repeat(1, 4, 4)
    x = torch.cat((x, y), dim=0)
    out_1 = self.c_5(x)
    out_2 = self.c_6(x)
    out_1 = out_1.squeeze(1).squeeze(1)
    out_2 = out_2.squeeze(1).squeeze(1)
    return out_1, out_2
    
  def generator(self, z, y):
    '''
    In: 128, 128, out: 64*64*3
    '''
    x = torch.cat(z, y, dim=0)
    x = self.fc_1(x)
    x = x.reshape(4, 4, 512)
    print(x.shape)
    x = self.bn_5(x)
    x = self.dc_1(x)
    x = self.bn_6(x)
    x = self.relu(x)
    x = self.dc_2(x)
    x = self.bn_7(x)
    x = self.relu(x)
    x = self.dc_3(x)
    x = self.bn_8(x)
    x = self.relu(x)
    x = self.dc_4(x)
    x = self.tanh(x)
    return x
  
  
  def discriminator(self, x, z, y):
    '''
    In: 64*64*3, 128, 128, out: 1
    '''
    y = y.repeat(1, 64, 64)
    x = torch.cat(x, y)
    x = self.c_7(x)
    x = self.leakyrelu(x)
    x = self.c_8(x)
    x = self.bn_9(x)
    x = self.leakyrelu(x)
    x = self.c_9(x)
    x = self.bn_10(x)
    x = self.leakyrelu(x)
    x = self.c_10(x)
    x = self.bn_11(x)
    x = self.leakyrelu(x)
    x = self.c_11(x)
    x = torch.cat(x, z, dim=0)
    x = self.fc_2(x)
    x = self.leakyrelu(x)
    return x
    
  
  
  def forward(self, x, y):
    
    return x
  
  
if __name__ == '__main__':
  x = torch.rand((3, 45, 60))
  y = torch.rand((128))
  z = torch.rand((128))
  NN = LGN()
  out = NN(x, y)
  print(out.shape)