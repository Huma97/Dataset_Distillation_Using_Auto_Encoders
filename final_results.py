import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def draw(files, path, baseline):

  arrays = []

  for f in files:
    with open(f, 'rb') as f:
      arrays.append(pickle.load(f))

  all_arrays = np.stack(arrays, axis=0)

  means = np.mean(all_arrays, axis=0)
  stds = np.std(all_arrays, axis=0)

  if not baseline:
    sns.set()
    plt.figure(figsize=(12, 10), dpi=80)

    epochs = [i for i, _ in enumerate(all_arrays[0])]

    y1 = means + stds
    y2 = means - stds

    plt.plot(epochs, means, 'r')
    plt.fill_between(epochs, y1, y2)

    plt.savefig(path)

  if not baseline:
    return round(means[-1], 3), round(stds[-1], 3)
  else:
    return round(means, 3), round(stds, 3)


def save_plots(path, baseline):
  os.chdir(path)
  train_loss_files = [f for f in os.listdir() if f.startswith('train_losses')]
  test_loss_files = [f for f in os.listdir() if f.startswith('test_losses')]
  final_train_loss = draw(train_loss_files, 'train_loss_means_and_stds.pdf', baseline)
  final_test_loss = draw(test_loss_files, 'test_loss_means_and_stds.pdf', baseline)
  train_accuracy_files = [f for f in os.listdir() if f.startswith('train_accuracies')]
  test_accuracy_files = [f for f in os.listdir() if f.startswith('test_accuracies')]
  final_train_accuracy = draw(train_accuracy_files, 'train_accuracy_means_and_stds.pdf', baseline)
  final_test_accuracy = draw(test_accuracy_files, 'test_accuracy_means_and_stds.pdf', baseline)

  return final_train_loss, final_test_loss, final_train_accuracy, final_test_accuracy

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--baseline', action='store_true')
  parser.add_argument('--conv_decoder', action='store_true')
  parser.add_argument('--gan_generator', action='store_true')
  parser.add_argument('--learn_labels', action='store_true')

  args = parser.parse_args()

  print(args)

  if args.conv_decoder:
    l_train, l_test, ac_train, ac_test = save_plots('results/pre_trained_decoder/conv_decoder/', args.baseline)
  elif args.gan_generator:
    if args.learn_labels:
      l_train, l_test, ac_train, ac_test = save_plots('results/pre_trained_decoder/gan_generator/learn_images_and_labels/', args.baseline)
    else:
      l_train, l_test, ac_train, ac_test = save_plots('results/pre_trained_decoder/gan_generator/learn_only_images/', args.baseline)
  elif args.baseline:
    l_train, l_test, ac_train, ac_test = save_plots('results/baseline/', args.baseline)
  else:
    if args.learn_labels:
      l_train, l_test, ac_train, ac_test = save_plots('results/no_decoder/learn_images_and_labels/', args.baseline)
    else:
      l_train, l_test, ac_train, ac_test = save_plots('results/no_decoder/learn_only_images/', args.baseline)

  print('Final mean train loss : {} \u00B1 {}'.format(l_train[0], l_train[1]))
  print('Final mean train accuracy : {} \u00B1 {}'.format(ac_train[0], ac_train[1]))
  print('Final mean test loss : {} \u00B1 {}'.format(l_test[0], l_test[1]))
  print('Final mean test accuracy : {} \u00B1 {}'.format(ac_test[0], ac_test[1]))
