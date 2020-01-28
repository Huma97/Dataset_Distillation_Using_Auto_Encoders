# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
from load_dataset import load_dataset
from get_random_images import get_all_images, get_random_images
from get_steps import get_steps
from data_iterator import prefetch_train_loader_iter
from decoders import Generator, ConvAutoencoder, LinearAutoencoder
from model import Net
from grads_calculation import forward, backward, accumulate_grad
from get_test_loss import get_metrics
from draw_distilled_images import draw_distilled_images
from draw_plots import draw_accuracy, draw_losses

from torch.optim import Adam, lr_scheduler

#from torch.autograd import Variable

def train_images(seed, CIFAR, distill_epochs, distill_steps, epochs,
                 learn_labels, fc_decoder, conv_decoder, gan_generator, pre_trained_decoder):

  torch.manual_seed(seed)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  train_loader, test_loader = load_dataset(CIFAR=CIFAR)

  if CIFAR:
    input_size = 32
    nc = 3
  else:
    input_size = 28
    nc = 1

  if conv_decoder:
    auto_encoder = ConvAutoencoder().to(device)
    if pre_trained_decoder:
      auto_encoder.load_state_dict(torch.load('weights/large_conv_auto_encoder.pth'))
  elif gan_generator:
    auto_encoder = Generator().to(device)
    if pre_trained_decoder:
      auto_encoder.load_state_dict(torch.load('weights/GAN-Generator.pth'))
  elif fc_decoder:
    auto_encoder = LinearAutoencoder().to(device)
    if pre_trained_decoder:
      auto_encoder.load_state_dict(torch.load('weights/linear_auto_encoder.pth'))

  decoder = None
  if fc_decoder or conv_decoder:
    encoder = auto_encoder.encoder
    decoder = auto_encoder.decoder
  elif gan_generator:
    decoder = auto_encoder.main
  if decoder == None:
    print('Learn images without using decoder')
  else:
    print(decoder)

  T = distill_epochs * distill_steps

  LABELS = []
  DATA = []
  params = []

  if not learn_labels:
    distill_label = torch.arange(10, dtype=torch.long, device=device).repeat(1, 1)
    distill_label = distill_label.t().reshape(-1)
    for _ in range(distill_steps):
      LABELS.append(distill_label)
  else:
    for _ in range(distill_steps):
      dl_array = np.random.normal(size=(10, 10))
      distill_label = torch.tensor(dl_array, dtype=torch.float, requires_grad=True, device=device)
      LABELS.append(distill_label)
      params.append(distill_label)

  if pre_trained_decoder:
    all_images = get_all_images()

  for _ in range(distill_steps):
    if gan_generator:
      z = np.random.uniform(-1, 1, size=(10, 100, 1, 1))
      z = torch.tensor(z, dtype=torch.float, requires_grad=True, device=device)
    elif fc_decoder:
      if pre_trained_decoder:
        images = get_random_images(all_images)
        images = images.to(device)
        images = images.view(images.size(0), -1)
        z = encoder(images)
        z = torch.tensor(z, dtype=torch.float, requires_grad=True, device=device)
      else:
        z = np.random.uniform(-1, 1, size=(10, 196))
        z = torch.tensor(z, dtype=torch.float, requires_grad=True, device=device)
    elif conv_decoder:
      if pre_trained_decoder:
        images = get_random_images(all_images)
        images = images.to(device)
        images = encoder(images)
        z = images.view(images.size(0), -1)
        z = torch.tensor(z, dtype=torch.float, requires_grad=True, device=device)
      else:
        z = np.random.uniform(-1, 1, size=(10, 196))
        z = torch.tensor(z, dtype=torch.float, requires_grad=True, device=device)
    else:
        z = torch.randn(10, nc, input_size, input_size, device=device, requires_grad=True)

    # if gan_generator:
    #   z = np.random.uniform(-1, 1, size=(10, 100, 1, 1))
    #   z = torch.from_numpy(z).float().to(device)
    # else:
    #   if not pre_trained_decoder:
    #     z = np.random.uniform(-1, 1, size=(10, 196))
    #     z = torch.from_numpy(z).float().to(device)
    #   else:
    #     images = get_random_images(all_images)
    #     images = images.to(device)
    #     if not conv_decoder:
    #       images = images.view(images.size(0), -1)
    #       z = encoder(images).to(device)
    #     else:
    #       images = encoder(images)
    #       z = images.view(images.size(0), -1).to(device)
    # z = Variable(z)
    # z.requires_grad = True

    DATA.append(z)
    params.append(z)

  raw_init_distill_lr = torch.tensor(0.02, device=device)
  raw_init_distill_lr = raw_init_distill_lr.repeat(T, 1)
  raw_distill_lrs = raw_init_distill_lr.expm1_().log_().requires_grad_()
  params.append(raw_distill_lrs)

  optimizer = Adam(params, lr=0.01, betas=(0.5, 0.999))
  scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

  for p in params:
    p.grad = torch.zeros_like(p)

  train_losses = []
  train_accuracies = []

  test_losses = []
  test_accuracies = []

  # losses = []

  for epoch, it, (rdata, rlabel) in prefetch_train_loader_iter(train_loader, epochs=epochs): # сэмплим батч настоящих данных

    if it == 0:
      scheduler.step()

    optimizer.zero_grad()

    rdata, rlabel = rdata.to(device, non_blocking=True), rlabel.to(device, non_blocking=True)

    # losses = []
    steps = get_steps(distill_epochs=distill_epochs, DATA=DATA, LABELS=LABELS,
                      raw_distill_lrs=raw_distill_lrs)  # для каждой эпохи distill_epochs получаем distill_data, distill_labels и lr

    grad_infos = []

    model = Net(nc).to(device)

    model.reset()

    l, saved = forward(model, rdata, rlabel, steps, fc_decoder, conv_decoder, gan_generator, decoder, learn_labels)  # l - loss на батче rdata, saved = (l, params, gws)
    # losses.append(l)
    grad_infos.append(backward(model, steps, saved, learn_labels))  # (datas, gdatas, lrs, glrs)
    accumulate_grad(grad_infos, learn_labels)

    # opt step
    #torch.nn.utils.clip_grad_norm(params, 5)
    optimizer.step()

    if it == len(train_loader) - 1:
      # _losses = torch.stack(losses, 0).mean()
      # loss = _losses.item()
      #train_losses.append(loss)

      train_loss, train_accuracy, test_loss, test_accuracy = get_metrics(nc, train_loader, test_loader, distill_epochs,
                                                                         DATA, LABELS, raw_distill_lrs, fc_decoder,
                                                                         conv_decoder, gan_generator, decoder, learn_labels)
      train_losses.append(train_loss)
      train_accuracies.append(train_accuracy)
      test_losses.append(test_loss)
      test_accuracies.append(test_accuracy)


      # losses = []
      print((
        'Epoch: {:4d} [{:7d}/{:7d} ({:2.0f}%)]\t Train Loss: {:.4f}\t \t Test Loss: {:.4f}\t'
      ).format(
        epoch, it * train_loader.batch_size, len(train_loader.dataset),
               100. * it / len(train_loader), train_loss, test_loss
      ))



  if CIFAR:
    path = 'results/CIFAR/'
  else:
    path = 'results/MNIST/'

  filename = path + 'steps.pkl'
  with open(filename, 'wb') as f:
    pickle.dump(steps, f)


  fig = draw_distilled_images(CIFAR, distill_epochs, DATA, LABELS, raw_distill_lrs, fc_decoder, conv_decoder,
                              gan_generator, decoder, learn_labels)

  if pre_trained_decoder:
    if conv_decoder:
      fig.savefig(path + 'pre_trained_decoder/conv_decoder/images.pdf')
    elif gan_generator:
      if learn_labels:
        fig.savefig(path + 'pre_trained_decoder/gan_generator/learn_images_and_labels/images.pdf')
      else:
        fig.savefig(path + 'pre_trained_decoder/gan_generator/learn_only_images/images.pdf')
    elif fc_decoder:
      fig.savefig(path+ 'pre_trained_decoder/fc_decoder/images.pdf')
  else:
    if conv_decoder:
      fig.savefig(path + 'random_decoder/conv_decoder/images.pdf')
    elif fc_decoder:
      fig.savefig(path + 'random_decoder/fc_decoder/images.pdf')
    else:
      if learn_labels:
        fig.savefig(path + 'no_decoder/learn_images_and_labels/images.pdf')
      else:
        fig.savefig(path + 'no_decoder/learn_only_images/images.pdf')

  print('Images_saved')

  draw_losses(train_losses=train_losses, test_losses=test_losses)

  if pre_trained_decoder:
    if conv_decoder:
      plt.savefig(path + 'pre_trained_decoder/conv_decoder/losses.pdf')
      filename_1 = path + 'pre_trained_decoder/conv_decoder/train_losses_{}.pkl'.format(seed)
      filename_2 = path + 'pre_trained_decoder/conv_decoder/test_losses_{}.pkl'.format(seed)
      with open(filename_1, 'wb') as f:
        pickle.dump(train_losses, f)
      with open(filename_2, 'wb') as f:
        pickle.dump(test_losses, f)
    elif gan_generator:
      if learn_labels:
        plt.savefig(path + 'pre_trained_decoder/gan_generator/learn_images_and_labels/losses.pdf')
        filename_1 = path + 'pre_trained_decoder/gan_generator/learn_images_and_labels/train_losses_{}.pkl'.format(seed)
        filename_2 = path + 'pre_trained_decoder/gan_generator/learn_images_and_labels/test_losses_{}.pkl'.format(seed)
        with open(filename_1, 'wb') as f:
          pickle.dump(train_losses, f)
        with open(filename_2, 'wb') as f:
          pickle.dump(test_losses, f)
      else:
        plt.savefig(path + 'pre_trained_decoder/gan_generator/learn_only_images/losses.pdf')
        filename_1 = path + 'pre_trained_decoder/gan_generator/learn_only_images/train_losses_{}.pkl'.format(seed)
        filename_2 = path + 'pre_trained_decoder/gan_generator/learn_only_images/test_losses_{}.pkl'.format(seed)
        with open(filename_1, 'wb') as f:
          pickle.dump(train_losses, f)
        with open(filename_2, 'wb') as f:
          pickle.dump(test_losses, f)
    elif fc_decoder:
      plt.savefig(path + 'pre_trained_decoder/fc_decoder/losses.pdf')
  else:
    if conv_decoder:
      plt.savefig(path + 'random_decoder/conv_decoder/losses.pdf')
    elif fc_decoder:
      plt.savefig(path + 'random_decoder/fc_decoder/losses.pdf')
    else:
      if learn_labels:
        plt.savefig(path + 'no_decoder/learn_images_and_labels/losses.pdf')
        filename_1 = path + 'no_decoder/learn_images_and_labels/train_losses_{}.pkl'.format(seed)
        filename_2 = path + 'no_decoder/learn_images_and_labels/test_losses_{}.pkl'.format(seed)
        with open(filename_1, 'wb') as f:
          pickle.dump(train_losses, f)
        with open(filename_2, 'wb') as f:
          pickle.dump(test_losses, f)
      else:
        plt.savefig(path + 'no_decoder/learn_only_images/losses.pdf')
        filename_1 = path + 'no_decoder/learn_only_images/train_losses_{}.pkl'.format(seed)
        filename_2 = path + 'no_decoder/learn_only_images/test_losses_{}.pkl'.format(seed)
        with open(filename_1, 'wb') as f:
          pickle.dump(train_losses, f)
        with open(filename_2, 'wb') as f:
          pickle.dump(test_losses, f)

  draw_accuracy(train_accuracy=train_accuracies, test_accuracy=test_accuracies)

  if pre_trained_decoder:
    if conv_decoder:
      plt.savefig(path + 'pre_trained_decoder/conv_decoder/accuracy.pdf')
      filename_1 = path + 'pre_trained_decoder/conv_decoder/train_accuracies_{}.pkl'.format(seed)
      filename_2 = path + 'pre_trained_decoder/conv_decoder/test_accuracies_{}.pkl'.format(seed)
      with open(filename_1, 'wb') as f:
        pickle.dump(train_accuracies, f)
      with open(filename_2, 'wb') as f:
        pickle.dump(test_accuracies, f)
    elif gan_generator:
      if learn_labels:
        plt.savefig(path + 'pre_trained_decoder/gan_generator/learn_images_and_labels/accuracy.pdf')
        filename_1 = path + 'pre_trained_decoder/gan_generator/learn_images_and_labels/train_accuracies_{}.pkl'.format(seed)
        filename_2 = path + 'pre_trained_decoder/gan_generator/learn_images_and_labels/test_accuracies_{}.pkl'.format(seed)
        with open(filename_1, 'wb') as f:
          pickle.dump(train_accuracies, f)
        with open(filename_2, 'wb') as f:
          pickle.dump(test_accuracies, f)
      else:
        plt.savefig(path + 'pre_trained_decoder/gan_generator/learn_only_images/accuracy.pdf')
        filename_1 = path + 'pre_trained_decoder/gan_generator/learn_only_images/train_accuracies_{}.pkl'.format(seed)
        filename_2 = path + 'pre_trained_decoder/gan_generator/learn_only_images/test_accuracies_{}.pkl'.format(seed)
        with open(filename_1, 'wb') as f:
          pickle.dump(train_accuracies, f)
        with open(filename_2, 'wb') as f:
          pickle.dump(test_accuracies, f)
    elif fc_decoder:
      plt.savefig(path + 'pre_trained_decoder/fc_decoder/accuracy.pdf')
  else:
    if conv_decoder:
      plt.savefig(path + 'random_decoder/conv_decoder/accuracy.pdf')
    elif fc_decoder:
      plt.savefig(path + 'random_decoder/fc_decoder/accuracy.pdf')
    else:
      if learn_labels:
        plt.savefig(path + 'no_decoder/learn_images_and_labels/accuracy.pdf')
        filename_1 = path + 'no_decoder/learn_images_and_labels/train_accuracies_{}.pkl'.format(seed)
        filename_2 = path + 'no_decoder/learn_images_and_labels/test_accuracies_{}.pkl'.format(seed)
        with open(filename_1, 'wb') as f:
          pickle.dump(train_accuracies, f)
        with open(filename_2, 'wb') as f:
          pickle.dump(test_accuracies, f)
      else:
        plt.savefig(path + 'no_decoder/learn_only_images/accuracy.pdf')
        filename_1 = path + 'no_decoder/learn_only_images/train_accuracies_{}.pkl'.format(seed)
        filename_2 = path + 'no_decoder/learn_only_images/test_accuracies_{}.pkl'.format(seed)
        with open(filename_1, 'wb') as f:
          pickle.dump(train_accuracies, f)
        with open(filename_2, 'wb') as f:
          pickle.dump(test_accuracies, f)

  print('Plots_saved')

  print('Final train accuracy : {}'.format(train_accuracies[-1]))
  print('Final test accuracy : {}'.format(test_accuracies[-1]))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=0, help='what seed of the random number generator to use')
  parser.add_argument('--CIFAR', action='store_true', help='use CIFAR10 as dataset')
  parser.add_argument('--distill_epochs', type=int, default=3, help='how many distill_epochs to train')
  parser.add_argument('--distill_steps', type=int, default=10, help='how many steps over one distill_epoch to train')
  parser.add_argument('--epochs', type=int, default=100, help='how many epochs to train')
  parser.add_argument('--learn_labels', action='store_true', help='whether to train labels or not')
  parser.add_argument('--fc_decoder', action='store_true', help='use fully-connected decoder')
  parser.add_argument('--conv_decoder', action='store_true', help='use convolutional decoder')
  parser.add_argument('--gan_generator', action='store_true', help='use GAN-generator as decoder')
  parser.add_argument('--pre_trained_decoder', action='store_true', help='whether to use pre_trained_encoder or random_encoder')
  args = parser.parse_args()

  print(args)

  train_images(seed=args.seed, CIFAR=args.CIFAR, distill_epochs=args.distill_epochs, distill_steps=args.distill_steps, epochs=args.epochs,
               fc_decoder = args.fc_decoder, conv_decoder=args.conv_decoder, gan_generator = args.gan_generator,
               pre_trained_decoder=args.pre_trained_decoder, learn_labels=args.learn_labels)

  # train_images(train_decoder=False, conv_decoder=True,
  #              pre_trained_decoder=True,
  #              distill_epochs=3, distill_steps=10, epochs=10)

