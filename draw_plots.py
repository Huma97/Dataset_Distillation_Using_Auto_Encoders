import seaborn as sns
import matplotlib.pyplot as plt

def draw_losses(train_losses, test_losses):

  sns.set()

  plt.figure(figsize=(12,10), dpi=80)

  epochs = [i for i, _ in enumerate(train_losses)]

  plt.plot(epochs, train_losses)
  plt.plot(epochs, test_losses)
  plt.gca().set(xlabel='Epoch number', ylabel='Loss')

  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)
  plt.title("LeNet on MNIST with random initialization", fontsize=22)
  plt.legend(['train_loss', 'test_loss'])

def draw_accuracy(train_accuracy, test_accuracy):
  sns.set()

  plt.figure(figsize=(18, 10), dpi=100)

  epochs = [i for i, _ in enumerate(test_accuracy)]

  plt.plot(epochs, train_accuracy)
  plt.plot(epochs, test_accuracy)
  plt.gca().set(xlabel='Epoch number', ylabel='Accuracy')

  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)

  plt.title("LeNet on MNIST with random initialization", fontsize=22)
  plt.legend(['train_accuracy', 'test_accuracy'])