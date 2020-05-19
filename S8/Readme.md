Task:

Go through this repository: https://github.com/kuangliu/pytorch-cifar (Links to an external site.)
Extract the ResNet18 model from this repository and add it to your API/repo. 
Use your data loader, model loading, train, and test code to train ResNet18 on Cifar10
Your Target is 85% accuracy. No limit on the number of epochs. Use default ResNet18 code (so params are fixed). 
Once done finish S8-Assignment-Solution
Assignment Solution: ResNet Model

Model Features:
Used GPU

ResNet Variant: ResNet18

Total Params: 11,173,962

Since the model was Overfitting I used L1 & L2

Also Trained the model a bit harder by adding few Image Augmentation Techniques like RandomRotation, HorizonatalFlip & Vertical Flip. Didn't make the mistake of adding all transformations together, but experimented with the first one, analysed the model performance, later added second and lastly included the third one.

Used CrossEntropyLoss() to calculate loss value.

Ran the model for 20 Epochs with

 * Highest Train Accuracy: 96.22%

 * Corresponding Test Accuracy: 89.16% 
