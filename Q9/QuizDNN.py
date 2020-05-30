
#----------------------------------------------_CfarResNet---------------------------------

from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F

# class DepthwiseSeparableConv2d(nn.Module):
#     def __init__(self, input, output, padding=0, bias=False):
#         super(DepthwiseSeparableConv2d, self).__init__()
#         self.depthwise = nn.Conv2d(input, input, kernel_size=3, padding=padding, groups=input, bias=bias)
#         self.pointwise = nn.Conv2d(input, output, kernel_size=1)

#     def forward(self, x):
#         out = self.depthwise(x)
#         out = self.pointwise(out)
#         return out

class MyNet(nn.Module): 
	
    def __init__(self, name="Model"):
        super(MyNet, self).__init__()
        self.name = name


# Conv Block1 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (3,3), padding = 1, bias = False), 
            nn.ReLU(),
            nn.BatchNorm2d(16)) # O/P:32   
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3,3), padding = 0, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32)) # O/P: 28  

        self.pool1 = nn.Sequential(nn.MaxPool2d((2,2))) # O/P: 14  
      
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 32 , out_channels = 16, kernel_size = (1,1), padding = 1 ,  bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(16)) # O/P: 14  

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3,3), padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32)) # O/P: 14        
        


        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64)) # O/P: 14 



        self.pool2 = nn.Sequential(nn.MaxPool2d((2,2))) # O/P: 7   

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = (1,1), padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(32)) # O/P:7  


        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64)) # O/P: 7 



        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels =64 , kernel_size = (3,3), padding = 0, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(64)) # O/P: 5 




    # GAP 
        self.gap = nn.Sequential(nn.AvgPool2d(5))
    
    #Fully connected layer
        self.fc = nn.Sequential(nn.Linear(64,10))

    #Dropout 
        self.drop = nn.Sequential(nn.Dropout(0.10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.drop(x)
        x = self.conv4(x)
        x = self.drop(x)
        x = self.conv5(x)
        x = self.drop(x)
        x = self.pool2(x)
        x = self.conv6(x)
        x = self.drop(x)

       
        x = self.conv7(x)
        x = self.drop(x)
        x = self.conv8(x)
        x = self.drop(x)

        
    
        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        x = self.fc(x)


        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
