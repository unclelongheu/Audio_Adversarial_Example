import torch
import torch.nn as nn

import numpy as np

from load_model import *


class Base(object):
    """
    Base class for aie.
    """

    def __init__(self, model, decay, epsilon, epoch, alpha, k, random_start, norm, device=None):
        if norm not in ['l2', 'lf']:
            raise Exception("Unsupported norm {}".format(norm))
        self.model = model
        self.decay = decay
        self.k = k
        self.epsilon = epsilon
        self.epoch = epoch
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = 2 * epsilon/epoch
        self.random_start = random_start
        self.norm = norm
        self.device = next(self.model[0].parameters()).device if device is None else device
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.cosine_similarity = nn.CosineSimilarity()

    def calculate_eta(self, rho):
        return torch.exp(self.k*(rho - 1)) / (1 + torch.exp(self.k * (rho - 1)))

    def adwa(self, x, grad_ave, grads, z):
        x_near_ave = x + z * self.alpha * grad_ave.sign()
        x_near_w =  x + z * self.alpha * grads[0].sign()
        x_near_s = x + z * self.alpha * grads[1].sign()

        logit_ave = [m(x_near_ave) for m in self.model]
        logit_ave = torch.stack(logit_ave, dim=0).mean(0)
        logit_w = self.model[0](x_near_w)
        logit_s = self.model[1](x_near_s)

        cs_w = self.cosine_similarity(logit_ave, logit_w)
        cs_s = self.cosine_similarity(logit_ave, logit_s)

        rho = cs_s / cs_w
        eta = self.calculate_eta(rho)
        eta = eta.view(-1, 1)
        return eta

    def get_adaptive_grad(self, x, label, delta):
        logits = [m(x) for m in self.model]
        logit = torch.stack(logits, dim=0).mean(0)
        uniform_loss = self.cross_entropy(logit, label)
        uniform_grad = self.get_grad(uniform_loss, delta, retain_graph=True)  #kn

        losses = [self.cross_entropy(logit, label) for logit in logits]
        grads = [self.get_grad(l, delta, retain_graph=True) for l in losses]  # 2kn
        eta = self.adwa(x, uniform_grad, grads, 0.5)
        eta = eta.clone().detach()
        logit = eta * logits[0] + (1-eta) * logits[1]
        loss = self.cross_entropy(logit, label)
        grad = self.get_grad(loss, delta)
        return grad

    def forward(self, data, label, **kwargs):

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            x = self.transform(data + delta, momentum=momentum)
            # Calculate the adaptive grad
            grad = self.get_adaptive_grad(x, label, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, momentum, self.alpha)

        return delta.detach()

    @staticmethod
    def get_grad(loss, delta, retain_graph=False, **kwargs):
        """
        The gradient calculation, which should be overridden when the attack need to tune the gradient (e.g., TIM, variance tuning, enhanced momentum, etc.)
        """
        return torch.autograd.grad(loss, delta, retain_graph=retain_graph, create_graph=False)[0]

    def get_momentum(self, grad, momentum, **kwargs):
        """
        The momentum calculation
        """
        return momentum * self.decay + grad / (grad.abs().mean(dim=(-1), keepdim=True))

    def init_delta(self, data, **kwargs):
        delta = torch.zeros_like(data).to(self.device)
        if self.random_start:
            if self.norm == 'lf':
                delta.uniform_(-self.epsilon, self.epsilon)
            else:
                delta.normal_(-self.epsilon, self.epsilon)
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=10).view(delta.size(0), 1, 1, 1)
                r = torch.zeros_like(data).uniform_(0, 1).to(self.device)
                delta *= r / n * self.epsilon
        # delta = frequency_clamp(delta)
        delta.requires_grad = True
        return delta

    def update_delta(self, delta, grad, alpha, **kwargs):
        if self.norm == 'lf':
            delta = torch.clamp(delta + alpha * grad.sign(), -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        # delta = frequency_clamp(delta + data) - data
        return delta

    @staticmethod
    def transform(data, **kwargs):
        return data

    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)