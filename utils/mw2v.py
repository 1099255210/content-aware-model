import torch
import torch.nn as nn
import gensim
import numpy as np
from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"
modelpath = './models/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(modelpath, binary=True)

class W2vNN(nn.Module):
  def __init__(self):
    super(W2vNN, self).__init__()
    self.fc = nn.Sequential(
      nn.Linear(300, 256),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
    ).to(device)
    
  def forward(self, x):
    x = self.fc(x)
    return x
    
def w2v_encode(
  word_list: List[str],
  verbose = False,
) -> torch.Tensor:
  
  '''
  Word list -> n * 300d -> 300d -> 128d
  '''
  
  vec_list = np.empty(shape=(len(word_list), 300), dtype=float)
  for idx, word in enumerate(word_list):
    vec_list[idx] = model.get_vector(word)
    
  sum_vec = np.sum(vec_list, axis=0)
  sum_vec = torch.from_numpy(sum_vec).to(torch.float32).to(device)

  w2vmodel = W2vNN().to(device)
  ret_vec = w2vmodel(sum_vec)
  if verbose:
    print('---------------- Word2vec result ----------------')
    torch.set_printoptions(threshold=10)
    print(ret_vec)
    print(f'Shape: {ret_vec.shape}')
    print('---------------- Word2vec resend ----------------')
  return ret_vec


if __name__ == '__main__':
  word_list = ['apple', 'technology', 'computer', 'simple']
  w2v_encode(word_list, verbose=True)