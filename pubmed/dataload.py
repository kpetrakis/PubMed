import config
from typing import Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader

def create_train_dev_dataloaders(data_path: str) -> Tuple[DataLoader, DataLoader]:
  '''
  data_path: the path to the data type. e.g. data/len256 for the 256 length tokenized text data. 
  returns: the train and dev Dataloaders
  '''
  train_inputs = torch.load(f"{data_path}/train/inputs.pt") 
  train_masks = torch.load(f"{data_path}/train/masks.pt") 
  train_labels = torch.load(f"{data_path}/train/labels.pt") 

  dev_inputs = torch.load(f"{data_path}/dev/inputs.pt") 
  dev_masks = torch.load(f"{data_path}/dev/masks.pt") 
  dev_labels = torch.load(f"{data_path}/dev/labels.pt") 

  g = torch.Generator().manual_seed(123456789)

  train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
  dev_dataset = TensorDataset(dev_inputs, dev_masks, dev_labels)

  train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, generator=g)
  dev_dataloader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=8, generator=g)

  return train_dataloader, dev_dataloader

def create_test_dataloader(data_path: str) -> DataLoader:
  '''
  data_path: the path to the data type. e.g. data/len256 for the 256 length tokenized text data. 
  returns: the Dataloader corresponding to the test data
  '''
  test_inputs = torch.load(f"{data_path}/test/inputs.pt") 
  test_masks = torch.load(f"{data_path}/test/masks.pt") 
  test_labels = torch.load(f"{data_path}/test/labels.pt") 

  g = torch.Generator().manual_seed(123456789)

  test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

  test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=8, generator=g)

  return test_dataloader
