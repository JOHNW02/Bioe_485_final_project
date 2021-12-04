import torch

class Dataset(torch.utils.data.Dataset):
  def __init__(self, X, y, indices):
    self.X = X
    self.y = y
    self.indices = indices
    
  def __len__(self):
    #y = self.y
    #length = y.shape[1]
    return len(self.indices)
  
  def __getitem__(self, sample_idx):
    Id = self.indices[sample_idx]
    return {
        "data": torch.from_numpy(self.X[Id]),
        "label": self.y[Id]
    }