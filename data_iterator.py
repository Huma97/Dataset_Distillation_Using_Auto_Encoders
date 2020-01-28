def prefetch_train_loader_iter(train_loader, epochs):
  train_iter = iter(train_loader)
  for epoch in range(epochs):
    niter = len(train_iter)
    prefetch_it = max(0, niter - 2)
    for it, val in enumerate(train_iter):
      # Prefetch (start workers) at the end of epoch BEFORE yielding
      if it == prefetch_it and epoch < epochs - 1:
        train_iter = iter(train_loader)
      yield epoch, it, val