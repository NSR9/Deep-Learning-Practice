

Run this network (Links to an external site.).  
Fix the network above:
    change the code such that it uses GPU
    change the architecture to C1C2C3C40 (basically 3 MPs)
    total RF must be more than 44
    one of the layers must use Depthwise Separable Convolution
    one of the layers must use Dilated Convolution
    use GAP (compulsory):- add FC after GAP to target #of classes (optional)
    achieve 80% accuracy, as many epochs as you want. Total Params to be less than 1M. 
    upload to Github
    Attempt S7-Assignment Solution
    
Important points regarding S7 Assignment submission:
S7 is the folder which contains S7-Assignment contents.
The S7_mainFile is the main and route file. This file imports the following files:
  1. execute.py
  2. image_tranformations.py
  3. cifar10_models.py
 All the data transformation and data loading and accuracy logs are embeded in S7_mainFile.ipynb 
    
    
    
Model Features:
Used GPU
Receptive Field = 62
Used Depthwise Separable Convolution
Used Dilated Convolution
Since the model was overfitting, I used Dropout of 10% with L1 & L2.
Ran the model for 15 Epochs
Got more than 80% accuracy after 9th epoch.
Implemented model checkpoint to save best model and also to save model.
Built a Image Classifier similar to ResNet Architecture(maintaining same number of channels for each convolution block).
Library Documentation:
1.image_transformations.py : Applies required image transformation to both Test & Train dataset aka PreProcessing.

2.cifar10_models.py: Consists of 2 models i.e seafarNet & cfarResNet(don't mind the names...)

3.execute.py: Scripts to Test & Train the model.



    
