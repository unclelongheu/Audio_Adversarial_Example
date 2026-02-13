from TransferAttacks.AdaptiveInterdomianEnsemble.base import *


class EmiAie(Base):
    def __init__(self, model, decay,  epsilon, epoch, alpha, k=3, num_sample=11, radius=7, random_start=False, norm='lf', device=None):
        super().__init__(model, decay, epsilon, epoch, alpha, k, random_start, norm, device)
        self.num_sample = num_sample
        self.radius = radius
        self.epoch = epoch
        self.decay = decay
        if alpha is not None:
            self.alpha = alpha

    def _transform(self, x, grad, **kwargs):
        """
        Admix the input for Admix Attack
        """
        factors = np.linspace(-self.radius, self.radius, num=self.num_sample)
        return torch.concat([x + factor * self.alpha * grad for factor in factors])

    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return self.cross_entropy(logits, label.repeat(self.num_sample))

    def get_adaptive_grad(self, x, label, delta):
        logits = [m(x) for m in self.model]
        logit = torch.stack(logits, dim=0).mean(0)
        uniform_loss = self.get_loss(logit, label)
        uniform_grad = self.get_grad(uniform_loss, x, retain_graph=True)

        losses = [self.get_loss(logit, label) for logit in logits]
        grads = [self.get_grad(l, x, retain_graph=True) for l in losses]
        eta = self.adwa(x, uniform_grad, grads, 0.5)
        eta = eta.clone().detach()
        logit = eta * logits[0] + (1-eta) * logits[1]
        loss = self.get_loss(logit, label)
        grad = self.get_grad(loss, delta)
        return grad

    def forward(self, data, label, **kwargs):

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        bar_grad = 0
        for _ in range(self.epoch):
            # Obtain the output
            x = self._transform(data + delta, grad=bar_grad)  # 22, 1, 22050
            # Calculate the adaptive grad
            grad = self.get_adaptive_grad(x, label, delta)

            bar_grad = grad / (grad.abs().mean(dim=-1, keepdim=True))

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, momentum, self.alpha)

        return delta.detach()