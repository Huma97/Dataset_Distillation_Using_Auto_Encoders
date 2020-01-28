import torch
from torchvision import datasets
import torchvision.transforms as transforms

def load_dataset(num_workers=0, batch_size=1024, CIFAR=False):

  # convert data to torch.FloatTensor

  if CIFAR:
    transform_list = [
      transforms.Pad(padding=4, padding_mode='reflect'),
      transforms.RandomCrop(32),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor()
    ]
    train_data = datasets.CIFAR10(root='data', train=True,
                                download=True, transform=transforms.Compose(transform_list))
    test_data = datasets.CIFAR10(root='data', train=False,
                               download=True, transform=transforms.Compose(transform_list))
  else:
  # choose the training and test datasets
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='data', train=True,
                                     download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False,
                                    download=True, transform=transform)


  # prepare data loaders
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

  test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

  return train_loader, test_loader
