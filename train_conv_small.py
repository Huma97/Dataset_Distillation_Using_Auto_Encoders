import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from model import Conv_Small
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)

num_workers = 0
batch_size = 20

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

model = Conv_Small().to(device)

print(model)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

n_epochs = 30

model.train()  # prep model for training

for epoch in range(n_epochs):

  train_loss = 0.0

  ###################
  # train the model #
  ###################
  for data, target in train_loader:

    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()

    output = model(data)

    loss = criterion(output, target)

    loss.backward()

    optimizer.step()

    train_loss += loss.item() * data.size(0)

  # print training statistics
  # calculate average loss over an epoch
  train_loss = train_loss / len(train_loader.dataset)

  # print(list(model.parameters())[1])

  print('Epoch: {} \tTraining Loss: {:.6f}'.format(
    epoch + 1,
    train_loss
  ))


test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()

for data, target in test_loader:

    data, target = data.to(device), target.to(device)

    output = model(data)

    loss = criterion(output, target)

    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(len(target)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
