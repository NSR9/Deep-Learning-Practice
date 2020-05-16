# Transformations 
import torchvision
from torchvision import transforms
from torchvision import datasets, transforms
import numpy as np
import torch
import matplotlib.pyplot as plt


class Transforms:
  """
  Helper class to create test and train transforms
  """
  def __init__(self, normalize=False, mean=None, stdev=None):
    if normalize and (not mean or not stdev):
      raise ValueError('mean and stdev both are required for normalize transform')
  
    self.normalize=normalize
    self.mean = mean
    self.stdev = stdev

  def test_transforms(self):
    transforms_list = [transforms.ToTensor()]
    if(self.normalize):
      transforms_list.append(transforms.Normalize(self.mean, self.stdev))
    return transforms.Compose(transforms_list)

  def train_transforms(self, pre_transforms=None, post_transforms=None):
    if pre_transforms:
      transforms_list = pre_transforms
    else:
      transforms_list = []
    transforms_list.append(transforms.ToTensor())

    if(self.normalize):
      transforms_list.append(transforms.Normalize(self.mean, self.stdev))
    if post_transforms:
      transforms_list.extend(post_transforms)
    return transforms.Compose(transforms_list)


class DataTransformandLoad:

  def TransformAndLoad(normalize = False, mean1=None, stdev1=None):
    mean1  = mean1
    stdev1 = stdev1

    
    trans = Transforms(normalize=True, mean=mean1, stdev=stdev1)

    train_transform = trans.train_transforms()
    test_transform = trans.test_transforms()
    train = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = train_transform)
    trainloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=4)

    test = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = test_transform)
    testloader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=False, num_workers=4)

    return trainloader,testloader
  