import random
import torch
from torchvision import datasets
import torchvision.transforms as transforms

def get_all_images():
  transform = transforms.ToTensor()
  train_data = datasets.MNIST(root='data', train=True,
                              download=True, transform=transform)

  random_loader = torch.utils.data.DataLoader(train_data, batch_size=60000)
  images = []

  data, labels = iter(random_loader).next()

  for label in range(10):
    x = [data[i] for i in range(len(data)) if labels[i] == label]
    images.append(x)

  return images


def get_random_images(images):
  random_images = []

  for i in range(10):
    random_images.append(random.choice(images[i]))

  random_images = torch.stack(random_images)

  return random_images