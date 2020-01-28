import torch.nn.functional as F

def get_steps(distill_epochs, DATA, LABELS, raw_distill_lrs):
  data_label_iterable = (x for _ in range(distill_epochs) for x in zip(DATA, LABELS))
  lrs = F.softplus(raw_distill_lrs).unbind()

  steps = []
  for (it_data, label), lr in zip(data_label_iterable, lrs):
    steps.append((it_data, label, lr))

  return steps