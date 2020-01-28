import torch
import torch.nn as nn
from torch.nn import init

from contextlib import contextmanager
from six import add_metaclass

class PatchModules(type):
    def __call__(cls, *args, **kwargs):
        r"""Called when you call ReparamModule(...) """
        net = type.__call__(cls, *args, **kwargs)

        # collect weight (module, name) pairs
        # flatten weights
        w_modules_names = []

        for m in net.modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None:
                    w_modules_names.append((m, n))

        net._weights_module_names = tuple(w_modules_names)

        ws = tuple(m._parameters[n].detach() for m, n in w_modules_names)

        assert len(set(w.dtype for w in ws)) == 1

        # reparam to a single flat parameter
        net._weights_numels = tuple(w.numel() for w in ws)
        net._weights_shapes = tuple(w.shape for w in ws)
        with torch.no_grad():
            flat_w = torch.cat([w.reshape(-1) for w in ws], 0)

        # remove old parameters, assign the names as buffers
        for m, n in net._weights_module_names:
            delattr(m, n)
            m.register_buffer(n, None)

        # register the flat one
        net.register_parameter('flat_w', nn.Parameter(flat_w, requires_grad=True))

        return net


@add_metaclass(PatchModules)
class ReparamModule(nn.Module):
    def _apply(self, *args, **kwargs):
        rv = super(ReparamModule, self)._apply(*args, **kwargs)
        return rv

    def get_param(self, clone=False):
        if clone:
            return self.flat_w.detach().clone().requires_grad_(self.flat_w.requires_grad)
        return self.flat_w

    @contextmanager
    def unflatten_weight(self, flat_w):
        ws = (t.view(s) for (t, s) in zip(flat_w.split(self._weights_numels), self._weights_shapes))
        for (m, n), w in zip(self._weights_module_names, ws):
            setattr(m, n, w)
        yield
        for m, n in self._weights_module_names:
            setattr(m, n, None)

    def forward_with_param(self, inp, new_w):
        with self.unflatten_weight(new_w):
            return nn.Module.__call__(self, inp)

    def __call__(self, inp):
        return self.forward_with_param(inp, self.flat_w)

    # make load_state_dict work on both
    # singleton dicts containing a flattened weight tensor and
    # full dicts containing unflattened weight tensors...
    def load_state_dict(self, state_dict, *args, **kwargs):
        if len(state_dict) == 1 and 'flat_w' in state_dict:
            return super(ReparamModule, self).load_state_dict(state_dict, *args, **kwargs)
        with self.unflatten_weight(self.flat_w):
            flat_w = self.flat_w
            del self.flat_w
            super(ReparamModule, self).load_state_dict(state_dict, *args, **kwargs)
        self.register_parameter('flat_w', flat_w)

    def reset(self, inplace=True):
        if inplace:
            flat_w = self.flat_w
        else:
            flat_w = torch.empty_like(self.flat_w).requires_grad_()
        with torch.no_grad():
            with self.unflatten_weight(flat_w):
                init_weights(self)
        return flat_w

def init_weights(net, init_type='xavier'):
  def init_func(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv') or classname == 'Linear':
      if getattr(m, 'bias', None) is not None:
        init.constant_(m.bias, 0.0)
      if getattr(m, 'weight', None) is not None:
        if init_type == 'normal':
          init.normal_(m.weight, 0.0, 1.0)
        elif init_type == 'xavier':
          init.xavier_normal_(m.weight, gain=1.0)
        elif init_type == 'xavier_unif':
          init.xavier_uniform_(m.weight, gain=1.0)
        elif init_type == 'default':
          if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        else:
          raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    elif 'Norm' in classname:
      if getattr(m, 'weight', None) is not None:
        m.weight.data.fill_(1)
      if getattr(m, 'bias', None) is not None:
        m.bias.data.zero_()

  net.apply(init_func)
  return net