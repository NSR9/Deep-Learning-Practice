S10 Assignment
Task: Assignment:

Pick your last code
Make sure  to Add CutOut to your code. It should come from your transformations (albumentations)
Use this repo: https://github.com/davidtvs/pytorch-lr-finder (Links to an external site.) 
    Move LR Finder code to your modules
    Implement LR Finder (for SGD, not for ADAM)
    Implement ![ReduceLROnPlateau](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau)
Find best LR to train your model
Use SDG with Momentum
Train for 50 Epochs. 
Show Training and Test Accuracy curves
Target 88% Accuracy.
Run GradCAM on the any 25 misclassified images. Make sure you mention what is the prediction and what was the ground truth label.
Submit
Assignment Solution: S10 Assignment Solution

S10 Assignment Solution file is not in this folder. Because of the repeated RELOAD error I couldn't save it inside. The assignment solution is outside S10 Folder, named S10_Assignment(GradCamFix).ipynb, the file link is mentioned above.

Model Features:
Used GPU as Device

ResNet Variant: ResNet18

Total Params: 11,173,962

Implemented MMDA, used Albumentations since it's easy to integrate with PyTorch.

Also Trained the model a bit harder by adding Image Augmentation Techniques like Rotation, HorizonatalFlip, Vertical Flip & Cutout.

Best Learning Rate: 0.009

Used CrossEntropyLoss() to calculate loss value.

Ran the model for 50 Epochs with

 * Highest Validation Accuracy: 88.91% (50th Epoch)
Implemented GradCam.

Plotted GradCam for 25 Misclassified Images

Library Documentation:
1.AlbTestTransforms.py : Applies required image transformation to Test dataset using Albumentations library.

2.AlbTrainTransforms.py : Applies required image transformation to Train dataset using Albumentations library.

3.resNet.py: Consists of ResNet variants

4.execute.py: Scripts to Test & Train the model.

5.DataLoaders.py: Scripts to load the dataloaders.

6.displayData.py: Consists of helper functions to plot images from dataset & misclassified images

7.rohan_library: Imports all the required libraries at once.

8.Gradcam: Consists of Gradcam class & other related functions.

9.LR Finder: LR finder using FastAI Approach

Misclassified Images


GradCam for Misclassified Images


Model Performance on Train & Test Data
 

Model Logs
Epoch    41: reducing learning rate of group 0 to 2.2877e-03.
Learning Rate = 0.0022876792454961 for EPOCH 42
EPOCH:  42
Loss=0.7903354167938232 Batch_id=390 Accuracy=92.59: 100%|██████████| 391/391 [01:07<00:00,  5.83it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.6438, Accuracy: 8859/10000 (88.59%)

Learning Rate = 0.0022876792454961 for EPOCH 43
EPOCH:  43
Loss=0.705917239189148 Batch_id=390 Accuracy=92.62: 100%|██████████| 391/391 [01:07<00:00,  5.82it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.3594, Accuracy: 8766/10000 (87.66%)

Learning Rate = 0.0022876792454961 for EPOCH 44
EPOCH:  44
Loss=0.8390583395957947 Batch_id=390 Accuracy=92.72: 100%|██████████| 391/391 [01:07<00:00,  5.82it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.6713, Accuracy: 8767/10000 (87.67%)

Epoch    44: reducing learning rate of group 0 to 2.0589e-03.
Learning Rate = 0.00205891132094649 for EPOCH 45
EPOCH:  45
Loss=0.7062127590179443 Batch_id=390 Accuracy=92.92: 100%|██████████| 391/391 [01:07<00:00,  5.83it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.6009, Accuracy: 8838/10000 (88.38%)

Learning Rate = 0.00205891132094649 for EPOCH 46
EPOCH:  46
Loss=0.8726668357849121 Batch_id=390 Accuracy=93.16: 100%|██████████| 391/391 [01:07<00:00,  5.82it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0911, Accuracy: 8864/10000 (88.64%)

Learning Rate = 0.00205891132094649 for EPOCH 47
EPOCH:  47
Loss=0.6966678500175476 Batch_id=390 Accuracy=93.10: 100%|██████████| 391/391 [01:07<00:00,  5.82it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.2770, Accuracy: 8837/10000 (88.37%)

Epoch    47: reducing learning rate of group 0 to 1.8530e-03.
Learning Rate = 0.001853020188851841 for EPOCH 48
EPOCH:  48
Loss=0.7848386764526367 Batch_id=390 Accuracy=93.52: 100%|██████████| 391/391 [01:07<00:00,  5.81it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.6072, Accuracy: 8834/10000 (88.34%)

Learning Rate = 0.001853020188851841 for EPOCH 49
EPOCH:  49
Loss=0.6624087691307068 Batch_id=390 Accuracy=93.53: 100%|██████████| 391/391 [01:07<00:00,  5.84it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.4730, Accuracy: 8874/10000 (88.74%)

Learning Rate = 0.001853020188851841 for EPOCH 50
EPOCH:  50
Loss=0.681330680847168 Batch_id=390 Accuracy=93.75: 100%|██████████| 391/391 [01:07<00:00,  5.83it/s]

Test set: Average loss: 0.4327, Accuracy: 8891/10000 (88.91%)

Epoch    50: reducing learning rate of group 0 to 1.6677e-03.
Learning Rate = 0.001667718169966657 for EPOCH 51
