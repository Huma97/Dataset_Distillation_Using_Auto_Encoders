from utils import ReparamModule
import torch.nn.functional as F
import torch.nn as nn

class Net(ReparamModule):
  def __init__(self, nc):
    self.nc = nc
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(self.nc, 20, 5, 1)
    self.conv2 = nn.Conv2d(20, 50, 5, 1)
    if self.nc == 3:
      self.fc1 = nn.Linear(12500, 500)
    else:
      self.fc1 = nn.Linear(4 * 4 * 50, 500)
    self.fc2 = nn.Linear(500, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2, 2)
    if self.nc == 3:
      x = x.view(-1, 12500)
    else:
      x = x.view(-1, 4 * 4 * 50)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)

    return x

class Conv_Small(ReparamModule):
  def __init__(self):
    super(Conv_Small, self).__init__()
    self.conv1 = nn.Conv2d(1, 96, 3, padding=1)
    self.batch_norm1 = nn.BatchNorm2d(96)
    self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
    self.batch_norm2 = nn.BatchNorm2d(96)
    self.conv3 = nn.Conv2d(96, 96, 3, padding=1)
    self.batch_norm3 = nn.BatchNorm2d(96)
    self.dropout = nn.Dropout(0.5)
    self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
    self.batch_norm4 = nn.BatchNorm2d(192)
    self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
    self.batch_norm5 = nn.BatchNorm2d(192)
    self.conv6 = nn.Conv2d(192, 192, 3, padding=1)
    self.batch_norm6 = nn.BatchNorm2d(192)
    self.conv7 = nn.Conv2d(192, 192, 3)
    self.batch_norm7 = nn.BatchNorm2d(192)
    self.conv8 = nn.Conv2d(192, 192, 1)
    self.batch_norm8 = nn.BatchNorm2d(192)
    self.conv9 = nn.Conv2d(192, 192, 1)
    self.batch_norm9 = nn.BatchNorm2d(192)

    self.fc = nn.Linear(192, 10)

  def forward(self, x):
    x = F.leaky_relu(self.conv1(x), 0.1)
    x = self.batch_norm1(x)
    x = F.leaky_relu(self.conv2(x), 0.1)
    x = self.batch_norm2(x)
    x = F.leaky_relu(self.conv3(x), 0.1)
    x = self.batch_norm3(x)
    x = F.max_pool2d(x, 2, 2)
    x = self.dropout(x)
    x = F.leaky_relu(self.conv4(x), 0.1)
    x = self.batch_norm4(x)
    x = F.leaky_relu(self.conv5(x), 0.1)
    x = self.batch_norm5(x)
    x = F.leaky_relu(self.conv6(x), 0.1)
    x = self.batch_norm6(x)
    x = F.max_pool2d(x, 2, 2)
    x = self.dropout(x)
    x = F.leaky_relu(self.conv7(x), 0.1)
    x = self.batch_norm7(x)
    x = F.leaky_relu(self.conv8(x), 0.1)
    x = self.batch_norm8(x)
    x = F.leaky_relu(self.conv9(x), 0.1)
    x = self.batch_norm9(x)
    x = F.avg_pool2d(x, 5)
    x = x.view(-1, 192)
    x = self.fc(x)

    return x