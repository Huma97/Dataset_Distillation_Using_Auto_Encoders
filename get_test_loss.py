from model import Net
from get_steps import get_steps
from loss import cross_entropy_with_probs
import numpy as np
import torch
import torch.nn as nn

def get_metrics(nc, train_loader, test_loader, distill_epochs, DATA, LABELS,
                raw_distill_lrs, fc_decoder, conv_decoder, gan_generator, decoder, learn_labels):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model_for_testing = Net(nc).to(device)

  model_for_testing.reset()

  w = model_for_testing.get_param()

  steps = get_steps(distill_epochs, DATA, LABELS, raw_distill_lrs)

  model_for_testing.train()

  for i, (data, label, lr) in enumerate(steps):
    if conv_decoder:
      data = data.view(-1, 4, 7, 7)
      decoder_output = decoder(data)
      output = model_for_testing.forward_with_param(decoder_output, w)
    elif fc_decoder or gan_generator:
      decoder_output = decoder(data)
      decoder_output = decoder_output.view(decoder_output.size(0), 1, 28, 28)
      output = model_for_testing.forward_with_param(decoder_output, w)
    else:
      output = model_for_testing.forward_with_param(data, w)
    if not learn_labels:
      loss = nn.CrossEntropyLoss()(output, label)
    else:
      loss = cross_entropy_with_probs(output, label)
    loss.backward(lr)

    with torch.no_grad():
      w.sub_(w.grad)
      w.grad = None

  model_for_testing.eval()  # prep model for evaluation

  # from torchvision import datasets
  # import torchvision.transforms as transforms
  # transform = transforms.ToTensor()
  # train_data = datasets.MNIST(root='data', train=True,
  #                             download=True, transform=transform)
  # test_data = datasets.MNIST(root='data', train=False,
  #                            download=True, transform=transform)
  # train_data.data = torch.cat((train_data.data, test_data.data[:5000]))
  # train_data.targets = torch.cat((train_data.targets, test_data.targets[:5000]))
  # test_data.data = torch.cat((test_data.data, train_data.data[:5000]))
  # test_data.targets = torch.cat((test_data.targets, train_data.targets[:5000]))
  # train_data.data = train_data.data[5000:]
  # train_data.targets = train_data.targets[5000:]
  # test_data.data = test_data.data[5000:]
  # test_data.targets = test_data.targets[5000:]
  # train_loader = torch.utils.data.DataLoader(train_data, batch_size=1024, num_workers=0, shuffle=True)
  # test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024,
  #                                           num_workers=0, shuffle=True)

  train_loss = 0.0
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))

  #count = 0
  for images, target in train_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    #if count == len(test_loader):
    #  break
    images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
    output = model_for_testing(images)
    # calculate the loss
    loss = nn.CrossEntropyLoss()(output, target)
    # update test loss
    train_loss += loss.item() * images.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(len(target)):
      label = target.data[i]
      class_correct[label] += correct[i].item()
      class_total[label] += 1
    #count += 1

  train_loss = train_loss / len(train_loader.dataset)
  #train_loss = train_loss / len(test_loader.dataset)

  train_accuracy = np.sum(class_correct) / np.sum(class_total)

  test_loss = 0.0
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))

  for images, target in test_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
    output = model_for_testing(images)
    # calculate the loss
    loss = nn.CrossEntropyLoss()(output, target)
    # update test loss
    test_loss += loss.item() * images.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(len(target)):
      label = target.data[i]
      class_correct[label] += correct[i].item()
      class_total[label] += 1

  test_loss = test_loss / len(test_loader.dataset)

  test_accuracy = np.sum(class_correct) / np.sum(class_total)

  return train_loss, train_accuracy, test_loss, test_accuracy