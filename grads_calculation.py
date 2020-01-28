import torch
import torch.nn as nn
from loss import cross_entropy_with_probs

def forward(model, rdata, rlabel, steps, fc_decoder, conv_decoder, gan_generator, decoder, learn_labels):
  # forward
  model.train()
  w = model.get_param()
  #     if random_encoder and train_decoder:
  #         decoder_w = G.get_param()
  #     elif pre_trained_encoder and train_decoder:
  #         decoder_w = decoder.get_param()
  #     cat_w = torch.cat([decoder_w, w])
  #     parameters = [cat_w]
  parameters = [w]
  gws = []

  for step_i, (data, label, lr) in enumerate(steps):  # (distill_data, distill_labels, lr)
    with torch.enable_grad():
      if conv_decoder:
        data = data.view(-1, 4, 7, 7)
        decoder_output = decoder(data)
        output = model.forward_with_param(decoder_output, w)
      elif fc_decoder or gan_generator:
        decoder_output = decoder(data)
        decoder_output = decoder_output.view(decoder_output.size(0), 1, 28, 28)
        output = model.forward_with_param(decoder_output, w)
      else:
        output = model.forward_with_param(data, w)
      if not learn_labels:
        loss = nn.CrossEntropyLoss()(output, label)
      else:
        loss = cross_entropy_with_probs(output, label)
    gw, = torch.autograd.grad(loss, w, lr, create_graph=True)

    with torch.no_grad():
      new_w = w.sub(gw).requires_grad_()
      #parameters.append(torch.cat([decoder_w, new_w]))
      parameters.append(new_w)
      gws.append(gw)
      w = new_w

  # final L
  model.eval()
  #output = model.forward_with_param(rdata, parameters[-1][decoder_w.shape[0]:])
  output = model.forward_with_param(rdata, parameters[-1])
  ll = nn.CrossEntropyLoss()(output, rlabel)
  return ll, (ll, parameters, gws)

def backward(model, steps, saved_for_backward, learn_labels):
  l, parameters, gws = saved_for_backward

  datas = []
  gdatas = []
  lrs = []
  glrs = []
  labels = []
  glabels = []

  dw, = torch.autograd.grad(l, (parameters[-1],))

  # backward
  model.train()
  # Notation:
  #   math:    \grad is \nabla
  #   symbol:  d* means the gradient of final L w.r.t. *
  #            dw is \d L / \dw
  #            dgw is \d L / \d (\grad_w_t L_t )
  # We fold lr as part of the input to the step-wise loss
  #
  #   gw_t     = \grad_w_t L_t       (1)
  #   w_{t+1}  = w_t - gw_t          (2)
  #
  # Invariants at beginning of each iteration:
  #   ws are BEFORE applying gradient descent in this step
  #   Gradients dw is w.r.t. the updated ws AFTER this step
  #      dw = \d L / d w_{t+1}
  for (data, label, lr), w, gw in reversed(list(zip(steps, parameters, gws))):
    # hvp_in are the tensors we need gradients w.r.t. final L:
    #   lr (if learning)
    #   data
    #   ws (PRE-GD) (needed for next step)
    #
    # source of gradients can be from:
    #   gw, the gradient in this step, whose gradients come from:
    #     the POST-GD updated ws

    hvp_in = [w]
    hvp_in.append(data)
    hvp_in.append(lr)
    if learn_labels:
      hvp_in.append(label)
    dgw = dw.neg()  # gw is already weighted by lr, so simple negation
    hvp_grad = torch.autograd.grad(
      outputs=(gw,),
      inputs=hvp_in,
      grad_outputs=(dgw,)
    )
    # Update for next iteration, i.e., previous step
    with torch.no_grad():
      # Save the computed gdata and glrs
      datas.append(data)
      gdatas.append(hvp_grad[1])
      lrs.append(lr)
      glrs.append(hvp_grad[2])
      if learn_labels:
        labels.append(label)
        glabels.append(hvp_grad[3])

      # Update for next iteration, i.e., previous step
      # Update dw
      # dw becomes the gradients w.r.t. the updated w for previous step
      dw.add_(hvp_grad[0])

  return datas, gdatas, lrs, glrs, labels, glabels

def accumulate_grad(grad_infos, learn_labels):
  bwd_out = []
  bwd_grad = []
  for datas, gdatas, lrs, glrs, labels, glabels in grad_infos:
    bwd_out += list(lrs)
    bwd_grad += list(glrs)
    for d, g in zip(datas, gdatas):
      d.grad.add_(g)
    if learn_labels:
      for l, g in zip(labels, glabels):
        l.grad.add_(g)
  if len(bwd_out) > 0:
    torch.autograd.backward(bwd_out, bwd_grad)