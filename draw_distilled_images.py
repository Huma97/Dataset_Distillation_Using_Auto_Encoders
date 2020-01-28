import torch
import numpy as np
import matplotlib.pyplot as plt
from get_steps import get_steps

def draw_distilled_images(CIFAR, distill_epochs, DATA, LABELS, raw_distill_lrs, fc_decoder,
                          conv_decoder, gan_generator, decoder, learn_labels):

  if CIFAR:
    nc = 3
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
    label_names = ('plane', 'car', 'bird', 'cat',
             'deer', 'dog', 'monkey', 'horse', 'ship', 'truck')
  else:
    nc = 1
    mean = 0.1307
    std = 0.3081
    label_names = list(range(10))

  steps = get_steps(distill_epochs, DATA, LABELS, raw_distill_lrs)

  steps = [(d.detach().cpu(), l.detach().cpu(), lr) for (d, l, lr) in steps]

  if isinstance(steps[0][0], torch.Tensor):
    np_steps = []
    for data, label, lr in steps:
      if conv_decoder:
        data = data.view(-1, 4, 7, 7)
        data = decoder.to('cpu')(data)
        np_data = data.detach().permute(0, 2, 3, 1).to('cpu').numpy()
      elif fc_decoder or gan_generator:
        data = decoder.to('cpu')(data)
        np_data = data.detach().view(data.size(0), 28, 28, 1).to('cpu').numpy()
      else:
        np_data = data.detach().permute(0, 2, 3, 1).to('cpu').numpy()
      np_label = label.detach().to('cpu').numpy()
      if lr is not None:
        lr = lr.detach().cpu().numpy()
      np_steps.append((np_data, np_label, lr))

  N = len(np_steps[0][0])
  nrows = 2
  grid = (nrows, np.ceil(N / float(nrows)).astype(int))
  # plt.rcParams["figure.figsize"] = (grid[1] * 1.5 + 1, nrows * 1.5 + 1)

  plt.close('all')
  fig, axes = plt.subplots(nrows=grid[0], ncols=grid[1])
  axes = axes.flatten()
  data, labels, _ = np_steps[-1]

  for n, (img, label, axis) in enumerate(zip(data, labels, axes)):
    if nc == 1:
      img = img[..., 0]
    img = (img * std + mean).clip(0, 1)

    axis.axis('off')
    if learn_labels:
      sorted_indices = np.argsort(label)[::-1]
      first = "{0}: {1}".format(label_names[sorted_indices[0]], '%.1f' % label[sorted_indices[0]])
      second = "{0}: {1}".format(label_names[sorted_indices[1]], '%.1f' % label[sorted_indices[1]])
      third = "{0}: {1}".format(label_names[sorted_indices[2]], '%.1f' % label[sorted_indices[2]])
      axis.set_title("{0}\n{1}\n{2}".format(first, second, third))
    else:
      axis.set_title('Label {}'.format(label_names[label]))
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=[0, 0, 1, 0.95])
    axis.imshow(img, interpolation='nearest', cmap='gray')

  return fig