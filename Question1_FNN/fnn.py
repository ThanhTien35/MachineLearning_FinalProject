import numpy as np
from techniques import l2_penalty, l2_gradient


def mse(y, prediction):
    return np.mean((prediction - y) ** 2)


# Cau 1b - Mang FNN tu cai dat bang NumPy
class FNN:
    def __init__(self, layer_sizes, seed=42):
        rng = np.random.default_rng(seed)
        self.weights = [
            rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out))
            for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.biases = [np.zeros((1, n)) for n in layer_sizes[1:]]

    def forward(self, X):
        activations, preactivations = [X], []
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = activations[-1] @ W + b
            A = Z if i == len(self.weights) - 1 else np.maximum(0, Z)
            preactivations.append(Z)
            activations.append(A)
        return activations, preactivations

    def predict(self, X):
        return self.forward(X)[0][-1]

    def backward(self, y, activations, preactivations, l2=0.0):
        dW = [None] * len(self.weights)
        db = [None] * len(self.biases)
        # Dao ham MSE tai tang dau ra, sau do lan truyen nguoc.
        delta = 2 * (activations[-1] - y) / y.size

        for i in reversed(range(len(self.weights))):
            dW[i] = activations[i].T @ delta + l2_gradient(self.weights[i], l2)
            db[i] = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.weights[i].T) * (preactivations[i - 1] > 0)
        return dW, db

    def update(self, dW, db, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * db[i]

    def objective(self, X, y, l2=0.0):
        return mse(y, self.predict(X)) + l2_penalty(self.weights, l2)

    def get_state(self):
        return ([w.copy() for w in self.weights], [b.copy() for b in self.biases])

    def set_state(self, state):
        weights, biases = state
        self.weights = [w.copy() for w in weights]
        self.biases = [b.copy() for b in biases]

    def fit(self, train_data, val_data, learning_rate, epochs, l2=0.0, stopper=None):
        X_train, y_train = train_data
        X_val, y_val = val_data
        history = {'train_loss': [], 'val_loss': [], 'objective': []}

        for epoch in range(1, epochs + 1):
            A, Z = self.forward(X_train)
            dW, db = self.backward(y_train, A, Z, l2)
            self.update(dW, db, learning_rate)

            train_loss = mse(y_train, self.predict(X_train))
            val_loss = mse(y_val, self.predict(X_val))
            objective = train_loss + l2_penalty(self.weights, l2)
            if not np.isfinite(objective) or not np.isfinite(val_loss):
                raise FloatingPointError('Non-finite loss. Check learning rate.')
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['objective'].append(objective)

            if stopper is not None and stopper.step(val_loss, self, epoch):
                break

        if stopper is not None:
            stopper.restore(self)
        return history
