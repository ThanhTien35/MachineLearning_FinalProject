import numpy as np


# Cau 1c - Early Stopping
class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.reference_loss = float('inf')
        self.best_epoch = 0
        self.best_state = None
        self.wait = 0

    def step(self, loss, model, epoch):
        # Luu mo hinh tot nhat; min_delta chi dieu khien bo dem patience.
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.best_state = model.get_state()

        if loss < self.reference_loss - self.min_delta:
            self.reference_loss = loss
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.set_state(self.best_state)


# Cau 1d - L2 Regularization (khong ap dung cho bias)
def l2_penalty(weights, strength):
    return 0.5 * strength * sum(np.sum(w ** 2) for w in weights)


def l2_gradient(weight, strength):
    return strength * weight
