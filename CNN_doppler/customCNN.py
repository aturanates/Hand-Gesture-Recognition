import torch
import torch.nn as nn
from torch.autograd import Variable

class CNNModule(nn.Module):
    def __init__(self):
        super(CNNModule,self).__init__()

        #Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=4,out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.LeakyReLU()

        #Max Pool1
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        #Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=4,out_channels=16,kernel_size=5,stride=1,padding=0)
        self.relu2 = nn.LeakyReLU()

        #Max Pool2
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        #Fully Connected

        self.fc1 = nn.Linear(32*4*4,3)
