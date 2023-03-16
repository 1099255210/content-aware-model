import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
  
    self.pad = nn.ZeroPad2d(padding=(9, 10, 2, 2, 0, 0))
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
    

  def forward(self, x, y):
    '''
    In: 64*64*3, 128, out: 128, 128
    '''
    x = self.pad(x)
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
    
    mean = self.c_5(x)
    std = self.c_6(x)
    mean = mean.squeeze(1).squeeze(1)
    std = std.squeeze(1).squeeze(1)
    return mean, std
   

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
  
    self.fc_1 = nn.Linear(in_features=256, out_features=8192)
    self.bn_5 = nn.BatchNorm1d(4)
    self.dc_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
    self.bn_6 = nn.BatchNorm1d(8)
    self.dc_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    self.bn_7 = nn.BatchNorm1d(16)
    self.dc_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    self.bn_8 = nn.BatchNorm1d(32)
    self.dc_4 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    self.relu = nn.ReLU()
    self.tanh = nn.Hardtanh(min_val=0, max_val=1)
    
  def forward(self, z, y):
    '''
    In: 128, 128, out: 64*64*3
    '''
    x = torch.cat((z, y), dim=0)
    
    x = self.fc_1(x)
    x = x.reshape(512, 4, 4)
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
    x = torch.round(x)
    return x
    
  
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    self.c_7 = nn.Conv2d(in_channels=131, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    self.c_8 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    self.bn_9 = nn.BatchNorm1d(16)
    self.c_9 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    self.bn_10 = nn.BatchNorm1d(8)
    self.c_10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    self.bn_11 = nn.BatchNorm1d(4)
    self.c_11 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(4, 4), stride=(1, 1))
    self.fc_2 = nn.Linear(in_features=256, out_features=1)
  
    
  def forward(self, x, z, y):
    '''
    In: 64*64*3, 128, 128, out: 1
    '''
    
    y = y.unsqueeze(1).unsqueeze(1)
    y = y.repeat(1, 64, 64)
    x = torch.cat((x, y))
    
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
    
    x = x.squeeze(1).squeeze(1)
    x = torch.cat((x, z), dim=0)
    x = self.fc_2(x)
    x = self.leakyrelu(x)
    return x
  
  
if __name__ == '__main__':
  x = torch.rand((3, 60, 45))
  y = torch.rand((128))
  z = torch.rand((128))
  encoder = Encoder()
  generator = Generator()
  discriminator = Discriminator()

  z_hat = encoder(x, y)
  x_tilde = generator(z, y)
  result = discriminator(x, z, y)

  # out = NN(x, y, z)
  # print(out)