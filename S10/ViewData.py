  import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
from google.colab import files
import data_transform as dt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
tl = dt.DataTransformandLoad()
channel_means = (0.5, 0.5, 0.5)
channel_stdevs = (0.5, 0.5, 0.5)
trainloader, testloader = tl.TransformAndLoad(mean1 = channel_means,stdev1 = channel_stdevs)



# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize this is make sure the image is visible, if this step is skipped then the resulting images have a dark portion
    npimg = img.numpy()   # converting image to numpy array format
    plt.imshow(np.transpose(npimg, (1, 2, 0)))    # transposing npimg array


# get some random training images
def getTrainImages():
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))



# def misclassifiedOnes(model):
#   model = model.to(device)
#   dataiter = iter(testloader) 
#   count = 0
#   fig = plt.figure(figsize=(13,13))

#   while count<25:
#       images, labels = dataiter.next()
#       images, labels = images.to(device), labels.to(device)
    
#       output = model(images) 
#       _, pred = torch.max(output, 1)   # convert output probabilities to predicted class
#       images = images.cpu().numpy() # conv images to numpy format

#       for idx in np.arange(128):
#         if pred[idx]!=labels[idx]:
#           ax = fig.add_subplot(5, 5, count+1, xticks=[], yticks=[])
#           count=count+1
#           ax.imshow(np.squeeze(images[idx]), cmap='cool')
#           ax.set_title("Pred-{} (Target-{})".format(str(pred[idx].item()), str(labels[idx].item())), color="Black")
#           if count==25:
#             break    

def misclassifiedOnes(model, testLoader, data,filename):

  #model: ModelName
  #data: Incorrect Classes in Test() of Test_Train class
  #filename: Pass on the filename with which you want to save misclassified images
  
  classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # classs names in the dataset

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  model = model.to(device)
  dataiter = iter(testLoader) 
  count = 0
  
  # Initialize plot
  fig = plt.figure(figsize=(13,13))
  
  row_count = -1
  fig, axs = plt.subplots(5, 5, figsize=(10, 10))
  fig.tight_layout()

  for idx, result in enumerate(data):

    # If 25 samples have been stored, break out of loop
    if idx > 24:
      break
        
    rgb_image = np.transpose(result['image'], (1, 2, 0)) / 2 + 0.5
    label = result['label'].item()
    prediction = result['prediction'].item()

    # Plot image
    if idx % 5 == 0:
      row_count += 1
    axs[row_count][idx % 5].axis('off')
    axs[row_count][idx % 5].set_title(f'Label: {classes[label]}\nPrediction: {classes[prediction]}')
    axs[row_count][idx % 5].imshow(rgb_image)
    
  # save the plot
  plt.savefig(filename)
  files.download(filename)