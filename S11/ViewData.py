import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import google.colab
from google.colab import files
import data_transform as dt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class_names = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
tl = dt.DataTransformandLoad()
channel_means = tl.channel_means
channel_stdevs = tl.channel_stdevs
trainloader, testloader = tl.TransformAndLoad()



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
    print(' '.join('%5s' % class_names[labels[j]] for j in range(4)))



   

def unnormalize(img):
  """
  Input: A Normal Image
  Output: Returns the unnormalized Image
  """
  img = img.numpy().astype(dtype=np.float32)
  for i in range(img.shape[0]):
    img[i] = (img[i]*channel_stdevs[i])+channel_means[i] # if not unnormalized then the resulting images will be dark and not visible
  return np.transpose(img, (1,2,0))



# Plot Class Specific Images
def class_images(dataiterator):
  """
  Returns a plot consisting of Random Images from all 10 classes.
  """
  num_classes = 10
  images, labels = iter(dataiterator).next()
  r, c = 10, 10
  n = 5
  fig = plt.figure(figsize=(14,14))
  fig.subplots_adjust(hspace=0.01, wspace=0.01)

  for i in range(num_classes):
    idx = np.random.choice(np.where(labels[:]==i)[0], n)
    ax = plt.subplot(r, c, i*c+1) # (10, 10, i*10+1)
    ax.text(-1.5, 0.5, class_names[i], fontsize=14)
    plt.axis('off')
    for j in range(1, n+1):
      plt.subplot(r, c, i*c+j+1) # (10, 10, i*10+j+1)
      plt.imshow(unnormalize(images[idx[j-1]]), interpolation='none')
      plt.axis('off')
  plt.show()



# Plot Random Image from Train Dataset
def plot_image(img):
    """
    Plots random images from training dataset
    """
    img = img / 2 + 0.5  # unnormalize this is make sure the image is visible, if this step is skipped then the resulting images have a dark portion
    npimg = img.numpy()   # converting image to numpy array format
    plt.imshow(np.transpose(npimg, (1, 2, 0)))    # transposing npimg array



# Misclassified ones
def misclassified_ones(model, testLoader, data,filename):
  """
  Arguments:
    model(str): ModelName
    testLoader: Data Loader for Test Images
    data(list): Incorrect Classes in Test() of Test_Train.py File
    filename(str): Return Image Save as
  """
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




# Correctly Classified Images
from google.colab import files
def correctly_classifed(model, testLoader, data, filename):
  """
  Arguments:
    model(str): ModelName
    testLoader: Data Loader for Test Images
    data(list): Incorrect Classes in Test() of Test_Train.py File
    filename(str): Return Image Save as
  """

  #model: ModelName
  #data: Correct Classes in Test() of Test_Train class
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
    if idx > 24:# If 25 samples have been stored, break out of loop
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



# Training & Validation Curves
def plot_curve(elements, title, y_label = 'Accuracy', Figsize = (8,8)):
    """
    elements: Contains Training and Testing variables of the Model like Accuracy or Loss
    title: Plot title
    y_label: Y-axis Label, Accuracy by default
    FigSize: Size of the Plot
    """
    with plt.style.context('fivethirtyeight'):
        fig = plt.figure(figsize=Figsize)
        ax = plt.subplot()
        for elem in elements:
            ax.plot(elem[0], label=elem[1])
            ax.set(xlabel='Epochs', ylabel=y_label)
            plt.title(title)
        ax.legend()
    plt.show()



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

# def misclassifiedOnes(model, testLoader, data,filename):

#   #model: ModelName
#   #data: Incorrect Classes in Test() of Test_Train class
#   #filename: Pass on the filename with which you want to save misclassified images
  
#   classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # classs names in the dataset

#   use_cuda = torch.cuda.is_available()
#   device = torch.device("cuda" if use_cuda else "cpu")
#   model = model.to(device)
#   dataiter = iter(testLoader) 
#   count = 0
  
#   # Initialize plot
#   fig = plt.figure(figsize=(13,13))
  
#   row_count = -1
#   fig, axs = plt.subplots(5, 5, figsize=(10, 10))
#   fig.tight_layout()

#   for idx, result in enumerate(data):

#     # If 25 samples have been stored, break out of loop
#     if idx > 24:
#       break
        
#     rgb_image = np.transpose(result['image'], (1, 2, 0)) / 2 + 0.5
#     label = result['label'].item()
#     prediction = result['prediction'].item()

#     # Plot image
#     if idx % 5 == 0:
#       row_count += 1
#     axs[row_count][idx % 5].axis('off')
#     axs[row_count][idx % 5].set_title(f'Label: {classes[label]}\nPrediction: {classes[prediction]}')
#     axs[row_count][idx % 5].imshow(rgb_image)
    
#   # save the plot
#   plt.savefig(filename)
#   files.download(filename)