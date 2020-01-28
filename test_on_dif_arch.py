import torch
import pickle
import torch.nn as nn
import numpy as np
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from model import Conv_Small
from load_dataset import load_dataset
from decoders import ConvAutoencoder, Generator
from loss import cross_entropy_with_probs

def get_results(path, epochs, learn_labels, conv_decoder, gan_generator):

  train_loader, test_loader = load_dataset()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  with open(path, 'rb') as f:
    steps = pickle.load(f)

  if conv_decoder:
    auto_encoder = ConvAutoencoder().to(device)
    auto_encoder.load_state_dict(torch.load('weights/large_conv_auto_encoder.pth'))
  elif gan_generator:
    auto_encoder = Generator().to(device)
    auto_encoder.load_state_dict(torch.load('weights/GAN-Generator.pth'))

  decoder = None
  if conv_decoder:
    encoder = auto_encoder.encoder
    decoder = auto_encoder.decoder
  elif gan_generator:
    decoder = auto_encoder.main
  if decoder == None:
    print('Learn without using decoder')
  else:
    print(decoder)

  model_for_testing = Conv_Small().to(device)

  model_for_testing.reset()

  w = model_for_testing.get_param()

  model_for_testing.train()

  for j in range(int(epochs/3)):
    for i, (data, label, lr) in enumerate(steps):
      if conv_decoder:
        data = data.view(-1, 4, 7, 7)
        decoder_output = decoder(data)
        output = model_for_testing.forward_with_param(decoder_output, w)
      elif gan_generator:
        decoder_output = decoder(data)
        decoder_output = decoder_output.view(decoder_output.size(0), 1, 28, 28)

        print('Max_value: {}'.format(torch.max(decoder_output)))
        print('Min_value: {}'.format(torch.min(decoder_output)))
        print('LR: {}'.format(lr))
        
        output = model_for_testing.forward_with_param(decoder_output, w)
      else:
        print('Max_value: {}'.format(torch.max(data)))
        print('Min_value: {}'.format(torch.min(data)))
        print('LR: {}'.format(lr))

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

  return test_loss, test_accuracy

def draw_losses(test_losses, epochs_list, learn_labels, conv_decoder, gan_generator):

  sns.set()

  plt.figure(figsize=(12, 10), dpi=80)

  plt.plot(epochs_list, test_losses)
  plt.gca().set(xlabel='Epoch number', ylabel='Test_Loss')

  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)
  plt.title("Conv_Small on distilled images", fontsize=22)

  if conv_decoder:
    if not learn_labels:
      path = 'steps/conv_decoder_learn_images_loss.pdf'
    else:
      path = 'steps/conv_decoder_learn_images_and_labels_loss.pdf'
  elif gan_generator:
    if not learn_labels:
      path = 'steps/gan_generator_learn_images_loss.pdf'
    else:
      path = 'steps/gan_generator_learn_images_and_labels_loss.pdf'
  else:
    if not learn_labels:
      path = 'steps/no_decoder_learn_images_loss.pdf'
    else:
      path = 'steps/no_decoder_learn_images_and_labels_loss.pdf'

  plt.savefig(path)


def draw_accuracies(test_accuracies, epochs_list, learn_labels, conv_decoder, gan_generator):

  sns.set()

  plt.figure(figsize=(12, 10), dpi=80)

  plt.plot(epochs_list, test_accuracies)
  plt.gca().set(xlabel='Epoch number', ylabel='Test_Accuracy')

  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)
  plt.title("Conv_Small on distilled images", fontsize=22)

  if conv_decoder:
    if not learn_labels:
      path = 'steps/conv_decoder_learn_images_accuracy.pdf'
    else:
      path = 'steps/conv_decoder_learn_images_and_labels_accuracy.pdf'
  elif gan_generator:
    if not learn_labels:
      path = 'steps/gan_generator_learn_images_accuracy.pdf'
    else:
      path = 'steps/gan_generator_learn_images_and_labels_accuracy.pdf'
  else:
    if not learn_labels:
      path = 'steps/no_decoder_learn_images_accuracy.pdf'
    else:
      path = 'steps/no_decoder_learn_images_and_labels_accuracy.pdf'

  plt.savefig(path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--learn_labels', action='store_true', help='whether to train labels or not')
  parser.add_argument('--conv_decoder', action='store_true', help='use convolutional decoder')
  parser.add_argument('--gan_generator', action='store_true', help='use GAN-generator as decoder')

  args = parser.parse_args()

  print(args)

  epochs_list = list(range(3, 102, 3))

  if args.conv_decoder:
    if not args.learn_labels:
      path = 'steps/conv_decoder_learn_images.pkl'
    else:
      path = 'steps/conv_decoder_learn_images_and_labels.pkl'
  elif args.gan_generator:
    if not args.learn_labels:
      path = 'steps/gan_generator_learn_images.pkl'
    else:
      path = 'steps/gan_generator_learn_images_and_labels.pkl'
  else:
    if not args.learn_labels:
      path = 'steps/no_decoder_learn_images.pkl'
    else:
      path = 'steps/no_decoder_learn_images_and_labels.pkl'


  test_losses = []
  test_accuracies = []
  for epochs in epochs_list:
    test_loss, test_accuracy = get_results(path, epochs, args.learn_labels, args.conv_decoder, args.gan_generator)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

  draw_losses(test_losses, epochs_list, args.learn_labels, args.conv_decoder, args.gan_generator)
  draw_accuracies(test_accuracies, epochs_list, args.learn_labels, args.conv_decoder, args.gan_generator)





