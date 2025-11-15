import numpy as np

class LogisticRegressionGD:
    """
    Logistic Regression using Gradient Descent.
    A simple ML-like model with fit(), predict(), predict_proba(), score(), etc.
    """

    def __init__(self, eta=0.01, epochs=1000, threshold=0.001, verbose=False):
        self.eta = eta
        self.epochs = epochs
        self.threshold = threshold
        self.verbose = verbose
        self.w = None
        self.b = None
        self.loss_history = []

    # ------------------------------------------------------------------
    @staticmethod
    def sigmoid(z):
        """Numerically stable sigmoid."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        m, n = X_train.shape
        self.w = np.zeros((n, 1))
        self.b = 0
        y_train = y_train.reshape(-1, 1)

        print(f"🚀 Training started | eta={self.eta}, epochs={self.epochs}")

        for epoch in range(self.epochs):
            # Forward step
            z = np.dot(X_train, self.w) + self.b
            y_hat = self.sigmoid(z)

            # Cross-entropy loss
            loss = - (1/m) * np.sum(
                y_train * np.log(y_hat + 1e-9) + (1 - y_train) * np.log(1 - y_hat + 1e-9)
            )
            self.loss_history.append(loss)

            # Gradient
            dw = (1/m) * np.dot(X_train.T, (y_hat - y_train))
            db = (1/m) * np.sum(y_hat - y_train)

            # Update parameters
            self.w -= self.eta * dw
            self.b -= self.eta * db

            # Verbose logging
            if self.verbose and epoch % 100 == 0:
                msg = f"[eta={self.eta}] Epoch {epoch:04d} | Loss={loss:.6f}"
                if X_val is not None and y_val is not None:
                    acc = self.score(X_val, y_val)
                    msg += f" | Val Accuracy={acc:.4f}"
                print(msg)

            # Early stopping
            if loss < self.threshold:
                if self.verbose:
                    print(f"✔️ Converged at epoch {epoch}, Loss={loss:.6f}")
                break

        print(f"🏁 Training completed after {epoch+1} epochs. Final Loss: {loss:.6f}")

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """Return predicted probability."""
        return self.sigmoid(np.dot(X, self.w) + self.b)

    # ------------------------------------------------------------------
    def predict(self, X, threshold=0.5):
        """Return class predictions (0 or 1)."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------------
    def score(self, X, y):
        """Return accuracy."""
        y_pred = self.predict(X)
        return np.mean(y_pred.flatten() == y.flatten())

    # ------------------------------------------------------------------
    def get_loss_history(self):
        return np.array(self.loss_history)

    # ------------------------------------------------------------------
    def get_params(self):
        """Return model hyperparameters (sklearn style)."""
        return {
            "eta": self.eta,
            "epochs": self.epochs,
            "threshold": self.threshold,
            "verbose": self.verbose
        }

    def set_params(self, **params):
        """Set model hyperparameters (sklearn style)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    def predict_with_debug(self, X, y_true=None, show_samples=10):
        """
        Show detailed predictions: y_true vs y_pred vs prob.
        Useful for debugging eta and epochs.
        """
        y_hat_proba = self.predict_proba(X).flatten()
        y_pred = self.predict(X).flatten()

        print("\n" + "="*65)
        print("                 PREDICTION DETAILS")
        print("="*65)
        print(f"{'Index':<6} {'y_true':<8} {'y_pred':<8} {'proba':<12} {'Correct?'}")
        print("-"*65)

        correct = 0
        for i in range(len(y_pred)):
            proba = y_hat_proba[i]
            pred = int(y_pred[i])

            if y_true is not None:
                true = int(y_true[i])
                ok = "✔️" if pred == true else "✖️"
                if pred == true:
                    correct += 1
            else:
                true = "?"

            if i < show_samples:
                print(f"{i:<6} {true:<8} {pred:<8} {proba:<12.4f} {ok if y_true is not None else ''}")

        if y_true is not None:
            acc = correct / len(y_pred)
            print("-"*65)
            print(f"Accuracy: {acc:.4f} ({correct}/{len(y_pred)})")
            print(f"Settings → eta={self.eta}, epochs={len(self.loss_history)}")

        print("="*65 + "\n")

        return y_pred, y_hat_proba
