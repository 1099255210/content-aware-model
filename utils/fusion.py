import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class FuseNN(nn.Module):
  def __init__(self):
    super(FuseNN, self).__init__()
    self.fc = nn.Sequential(
      nn.Linear(288, 256),
      nn.ReLU(),
      nn.Linear(256, 128)
    )
    
  def forward(self, x) -> torch.Tensor:
    x = self.fc(x)
    return x
  
  
model = FuseNN().to(device)

def fusion(img_vec:torch.Tensor, text_vec:torch.Tensor, attr_vec:torch.Tensor, verbose=False) -> torch.Tensor:
  '''
  128d + 128d + 32d = 128d
  '''
  
  x = torch.cat((img_vec, text_vec, attr_vec), dim=0)
  ret_vec = model(x)
  
  if verbose:
    print('---------------- MulModal result ----------------')
    torch.set_printoptions(threshold=10)
    print(ret_vec)
    print(f'Shape: {ret_vec.shape}')
    print('---------------- MulModal resend ----------------')
  return ret_vec
  
