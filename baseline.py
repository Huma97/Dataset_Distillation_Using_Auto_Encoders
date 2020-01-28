from get_random_images import *
import argparse
import pickle
from load_dataset import load_dataset
from model import Net
from get_steps import get_steps
import numpy as np
import torch
import torch.nn as nn

def get_baseline(seed, distill_epochs, distill_steps):

  torch.manual_seed(seed)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  train_loader, test_loader = load_dataset()

  T = distill_epochs * distill_steps
  DATA = []
  LABELS = []
  all_images = get_all_images()

  distill_label = torch.arange(10, dtype=torch.long, device=device).repeat(1, 1)
  distill_label = distill_label.t().reshape(-1)

  for _ in range(distill_steps):
    images = get_random_images(all_images)
    DATA.append(images)
    LABELS.append(distill_label)

  lr = torch.tensor(0.02, device=device)
  lr = lr.repeat(T, 1)
  lrs = lr.expm1_().log_()

  model_for_testing = Net().to(device)

  model_for_testing.reset()

  w = model_for_testing.get_param()

  steps = get_steps(distill_epochs, DATA, LABELS, lrs)

  model_for_testing.train()

  for i, (data, label, lr) in enumerate(steps):

    output = model_for_testing.forward_with_param(data, w)
    loss = nn.CrossEntropyLoss()(output, label)
    loss.backward(lr)
    with torch.no_grad():
      w.sub_(w.grad)
      w.grad = None

  model_for_testing.eval()  # prep model for evaluation

  train_loss = 0.0
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))

  for images, target in train_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
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

  train_loss = train_loss / len(train_loader.dataset)

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

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=0, help='what seed of the random number generator to use')
  parser.add_argument('--distill_epochs', type=int, default=3, help='how many distill_epochs to train')
  parser.add_argument('--distill_steps', type=int, default=10, help='how many steps over one distill_epoch to train')

  args = parser.parse_args()

  print(args)

  train_loss, train_accuracy, test_loss, test_accuracy = get_baseline(seed=args.seed, distill_epochs=args.distill_epochs, distill_steps=args.distill_steps)

  filename_1 = 'results/baseline/train_losses_{}.pkl'.format(args.seed)
  filename_2 = 'results/baseline/test_losses_{}.pkl'.format(args.seed)
  filename_3 = 'results/baseline/train_accuracies_{}.pkl'.format(args.seed)
  filename_4 = 'results/baseline/test_accuracies_{}.pkl'.format(args.seed)
  with open(filename_1, 'wb') as f:
    pickle.dump(train_loss, f)
  with open(filename_2, 'wb') as f:
    pickle.dump(test_loss, f)
  with open(filename_3, 'wb') as f:
    pickle.dump(train_accuracy, f)
  with open(filename_4, 'wb') as f:
    pickle.dump(test_accuracy, f)

  print('Final train loss : {}'.format(train_loss))
  print('Final train accuracy : {}'.format(train_accuracy))
  print('Final test loss : {}'.format(test_loss))
  print('Final test accuracy : {}'.format(test_accuracy))

