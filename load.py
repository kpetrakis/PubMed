import torch
from torch.utils.data import TensorDataset


inputs = torch.load('data/len256/train/inputs.pt') 
labels = torch.load('data/len256/train/labels.pt') 
masks = torch.load('data/len256/train/masks.pt') 

print(inputs.shape, labels.shape, masks.shape)
print(inputs.device)


dataset = TensorDataset(inputs, labels, masks)
print(len(dataset))