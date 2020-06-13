# Transformations 
import torchvision
import torch
import albumentations
from albumentations import pytorch as pyt
from albumentations import core as c
from albumentations import augmentations as aug
import Dataloaders
from Dataloaders import DataLoadersClass 
import Albumentations
from Albumentations import albTransforms


# class Transforms:
#   """
#   Helper class to create test and train transforms
#   """
#   def __init__(self, normalize=False, mean=None, stdev=None):
#     if normalize and (not mean or not stdev):
#       raise ValueError('mean and stdev both are required for normalize transform')
  
#     self.normalize=normalize
#     self.mean = mean
#     self.stdev = stdev

#   def test_transforms(self):
#     transforms_list = [pyt.transforms.ToTensor()]
#     if(self.normalize):
#       transforms_list.append(aug.transforms.Normalize(self.mean, self.stdev,always_apply=True))
#     return c.composition.Compose(transforms_list)







#   def train_transforms(self, means=None, pre_transforms=None, post_transforms=None):
#     channel_means = means
  
#    #imaage augmentation not data aug
    
#     #fillMeans = (np.array(channel_means)*255).astype(np.uint8)
#       #transforms_list = [transforms.RandomRotation((-15.0, 15.0), fill=tuple(fillMeans)), transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.5)]
#     transforms_list = [aug.transforms.HorizontalFlip(p=0.5)] 
#     transforms_list.append(
#         [
         
#          aug.transforms.Normalize(self.mean, self.stdev,always_apply=True),
#          pyt.transforms.ToTensor()]
#       )
#     print("transformation done")
#     print(transforms_list)

#     # if(self.normalize):
#     #   transforms_list.append(aug.transforms.Normalize(self.mean, self.stdev))
#     # if post_transforms:
#     #   transforms_list.extend(post_transforms)
#     print(c.composition.Compose(transforms_list))
#     return c.composition.Compose(transforms_list)


class DataTransformandLoad:
  def __init__(self):
    
    self.channel_means = (0.5, 0.5, 0.5)
    self.channel_stdevs = (0.5, 0.5, 0.5) 
  def TransformAndLoad(self,normalize = False):
    dl = DataLoadersClass()
    
    
    mean1  = self.channel_means
    stdev1 = self.channel_stdevs
    #use_cuda = torch.cuda.is_available()
    
    
    #trans = Transforms(normalize=True, mean=mean1, stdev=stdev1)
    #dataloader_args = dict(batch_size=128, shuffle=True, num_workers=4, pin_memory = True)if use_cuda else dict( batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    train_transform = albTransforms(train=True)
    test_transform = albTransforms()
    
    train = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = train_transform)
    trainloader = dl.dataLoader(train)

    test = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = test_transform)
    testloader = dl.dataLoader(test)
    return trainloader,testloader
  