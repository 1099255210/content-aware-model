import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class fuseNN(nn.Module):
  def __init__(self):
    super(fuseNN, self).__init__()
    self.fc = nn.Sequential(
      nn.Linear(288, 256),
      nn.ReLU(),
      nn.Linear(256, 128)
    )
    
  def forward(self, x) -> torch.Tensor:
    x = self.fc(x)
    return x
  

def fusion(img_vec:torch.Tensor, text_vec:torch.Tensor, attr_vec:torch.Tensor, verbose=False) -> torch.Tensor:
  x = torch.cat((img_vec, text_vec, attr_vec), dim=0)
  model = fuseNN().to(device)
  ret_vec = model(x)
  
  if verbose:
    print('---------------- MulModal result ----------------')
    torch.set_printoptions(threshold=10)
    print(ret_vec)
    print(f'Shape: {ret_vec.shape}')
    print('---------------- MulModal resend ----------------')
  return ret_vec
  
