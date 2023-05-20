import torch.nn as nn
import torch.nn.functional as F
import torch

class cnn(torch.nn.Module):

    def __init__(self, input_size=8,  output_size=1 , pre_days=7, feature_size=10):


        super().__init__()

        self.output_size = output_size

        self.conv1 = nn.Conv2d(1, 1, (3, 5))
        self.conv2 = nn.Conv2d(1, 1, (5, 5))
        self.conv3 = nn.Conv2d(1, 1, (3, 5))
        self.dim = (pre_days - 6) * (feature_size - 8)
        self.fc1 = nn.Linear(self.dim, 32)  # flattening.


        self.fc3 = nn.Linear(32, self.output_size)  #


    def forward(self, x):


        x = torch.unsqueeze(x, 1)
        x = F.leaky_relu(self.conv1(x))#leaky_relu
        x = F.leaky_relu(self.conv2(x))


        x = x.reshape(x.shape[0], -1)
        x =self.fc1(x)
        x =self.fc3(x)
        out = x


        return out