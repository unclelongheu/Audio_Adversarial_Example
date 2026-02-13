from TransferAttacks.AdaptiveInterdomianEnsemble.base import *


class MiAie(Base):
    """
    ENS Attack
    Delving into Transferable Adversarial Examples and Black-box Attacks (ICLR 2017)(https://arxiv.org/abs/1611.02770)
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.
    """

    def __init__(self, model, decay, epsilon, epoch, alpha, k=3, random_start=False, norm='lf', device=None):
        super().__init__(model, decay, epsilon, epoch, alpha, k, random_start, norm, device)
        self.epoch = epoch
        self.decay = decay
        if alpha is not None:
            self.alpha = alpha


