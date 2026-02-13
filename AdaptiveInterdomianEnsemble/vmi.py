from TransferAttacks.AdaptiveInterdomianEnsemble.base import *


class VmiAie(Base):
    """
    VMI-FGSM Attack
    'Enhancing the transferability of adversarial attacks through variance tuning (CVPR 2021)'(https://arxiv.org/abs/2103.15571)

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, beta=1.5, num_neighbor=20, epoch=10, decay=1.

    """

    def __init__(self, model, decay, epsilon, epoch, alpha, k=3, beta=1.5, num_neighbor=20,
                 random_start=False, norm='lf', device=None):
        super().__init__(model, decay, epsilon, epoch, alpha, k, random_start, norm, device)
        self.radius = beta * epsilon
        self.epoch = epoch
        self.decay = decay
        self.num_neighbor = num_neighbor
        if alpha is not None:
            self.alpha = alpha

    def get_variance(self, data, delta, label, cur_grad, momentum, **kwargs):
        """
        Calculate the gradient variance
        """
        grad = 0
        for _ in range(self.num_neighbor):
            # Obtain the output
            # This is inconsistent for transform!
            x_near = self.transform(
                data + delta + torch.zeros_like(delta).uniform_(-self.radius, self.radius).to(self.device),
                momentum=momentum)
            logits = [m(x_near) for m in self.model]
            logit = torch.stack(logits, dim=0).mean(0)

            # Calculate the loss
            loss = self.cross_entropy(logit, label)

            # Calculate the gradients
            grad += self.get_grad(loss, delta)

        return grad / self.num_neighbor - cur_grad

    def forward(self, data, label, **kwargs):

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum, variance = 0, 0
        for _ in range(self.epoch):
            x = self.transform(data + delta, momentum=momentum)
            # Calculate the adaptive grad
            grad = self.get_adaptive_grad(x, label, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad + variance, momentum)

            # Calculate the variance
            variance = self.get_variance(data, delta, label, grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, momentum, self.alpha)

        return delta.detach()

