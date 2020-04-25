5 Assignment
Model 1

File Link

Target: MNIST Digit Classifier with <10k Parameters.

Strategy: Restrict max number of channels to 32.

Model Info:
DropOut = 0.1(used once).
BatchNormalization after every Conv layer.

Result:
Total Parameters: 7,216
Highest Train Accuracy: 98.83%
Highest Test Accuracy: 98.55%
Corresponding Epoch: 14

Analysis:
The model is Overfitting.
Test & Train Accuracies can be improved.

Model 2


Target: No Overfitting but a Simple model.

Strategy: Use Data Augmentation technique, Rotating Input images by 5-7 Degrees to improve Test Accuracy. Use Dropout of 10% to avoid Overfitting.
Model Info:
DropOut = 0.1(used once).
BatchNormalization after every Conv layer.

Result:
Total Parameters: 8,622
Highest Train Accuracy: 98.68%
Highest Test Accuracy: 99.01%
Corresponding Epoch: 13

Analysis:
Not an Overfit Model. Thanks to Data Augmentation.
Small gap b/w Train and Test Accuracy seems to indicate Underfitting nature of the model.
Test Accuracies can be improved like the model can be pushed further.


Model 3



Target: Improve Test and Train Accuracy.

Strategy: Convolve deeper upto channel size of 6x6.
Model Info:
DropOut = 0.1(used twice).
BatchNormalization after every Conv layer.

Result:
Total Parameters: 8,870
Highest Train Accuracy: 98.97%
Highest Test Accuracy: 99.36%
Corresponding Epoch: 14

Analysis:
Both Test & Train Accuracies have improved.
The model is underfitting a bit so I guess I should remove a DropOut Layer.
Got stuck in 99.1 - 99.3 loop. Couldn't push forward with this architecture, so try convolving till channel size equals 5x5.


Model 4

File Link

Target: Improve both the Train & Test Accuracies(>99.2%)

Strategy: Increase number of channels and also convolve till 5x5.
Model Info:
DropOut = 0.05(used once).
BatchNormalization after every Conv layer.

Result:
Total Parameters: 9,688
Highest Train Accuracy: 99.31%(15th Epoch)
Highest Test Accuracy: 99.40%(15th Epoch)

Analysis:
Not an Overfitting Model.
Increase in Test Accuracy with 5 Epochs have >99.25%
Consistent model with overall accuracy of 99.3% change.
Loss goes


Model 5

File Link

Target: Improve the model results i.e Get Model Accuracy >= 99.4% on a consistent basis & Avoid Overfitting.

Strategy: Experiment with different LR Schedules especially with ReduceLROnPlateau & StepLR and for overfitting slightly increase the Dropout value.
Model Info:
DropOut = 0.1(used once).
BatchNormalization after every Conv layer.

Result:
Total Parameters: 9,108
Highest Train Accuracy: 99.40%
Highest Test Accuracy: 99.48%
Corresponding Epoch: 15

Analysis:
The model has achieved 99.4% of accuracy on consistent basis( in 4 Epochs).
Now the model is not Overfitting and all cheers to DropOut.
Also I have a another version() where in I have achieved 99.4% in 8 Epochs.
