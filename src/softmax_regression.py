# models/softmax_regression_gd.py
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Union
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


class SoftmaxRegressionGD:
    """
    Softmax Regression (Multinomial Logistic Regression) trained with Gradient Descent.

    Parameters
    ----------
    eta : float, default 0.01
        Learning rate.
    epochs : int, default 1000
        Maximum number of training epochs.
    threshold : float, default 1e-3
        Early stopping threshold for loss.
    verbose : bool, default False
        Print training progress every 100 epochs.

    Attributes
    ----------
    w : ndarray of shape (n_features, n_classes)
        Weight matrix.
    b : ndarray of shape (1, n_classes)
        Bias vector.
    loss_history : list
        Training loss per epoch.
    n_classes : int
        Number of classes.
    classes_ : ndarray
        Unique class labels.

    References
    ----------
    .. [1] Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
    .. [2] Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
    """

    def __init__(
        self,
        eta: float = 0.01,
        epochs: int = 1000,
        threshold: float = 1e-3,
        verbose: bool = False
    ):
        self.eta = eta
        self.epochs = epochs
        self.threshold = threshold
        self.verbose = verbose

        self.w = None
        self.b = None
        self.loss_history = []
        self.n_classes = None
        self.classes_ = None

    # ===================================================================
    # Static Methods
    # ===================================================================
    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable softmax function.

        Parameters
        ----------
        z : ndarray of shape (n_samples, n_classes)
            Linear scores.

        Returns
        -------
        probs : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)  # subtract max for stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @staticmethod
    def _one_hot_encode(y: np.ndarray, n_classes: int) -> np.ndarray:
        """
        Convert integer labels to one-hot encoding.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
        n_classes : int

        Returns
        -------
        onehot : ndarray of shape (n_samples, n_classes)
        """
        return np.eye(n_classes)[y.astype(int)]

    # ===================================================================
    # Core Methods
    # ===================================================================
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'SoftmaxRegressionGD':
        """
        Train the model using gradient descent.

        Parameters
        ----------
        X_train : ndarray of shape (n_samples, n_features)
        y_train : ndarray of shape (n_samples,)
        X_val, y_val : optional
            Validation set for accuracy monitoring.

        Returns
        -------
        self
        """
        X_train = np.array(X_train, dtype=np.float64)
        y_train = np.array(y_train, dtype=np.int64)

        if X_val is not None:
            X_val = np.array(X_val, dtype=np.float64)
            y_val = np.array(y_val, dtype=np.int64)

        m, n = X_train.shape
        self.classes_ = np.unique(y_train)
        self.n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        # Map labels to 0,1,...,K-1
        y_train_idx = np.array([class_to_idx[c] for c in y_train])
        y_train_onehot = self._one_hot_encode(y_train_idx, self.n_classes)

        # Initialize parameters
        self.w = np.zeros((n, self.n_classes))
        self.b = np.zeros((1, self.n_classes))
        self.loss_history = []

        print(f"Starting Softmax training | η={self.eta}, max_epochs={self.epochs}")

        for epoch in range(self.epochs):
            # Forward pass
            z = X_train @ self.w + self.b
            y_hat = self._softmax(z)

            # Cross-entropy loss
            loss = -np.mean(np.sum(y_train_onehot * np.log(y_hat + 1e-9), axis=1))
            self.loss_history.append(loss)

            # Gradients
            error = y_hat - y_train_onehot
            dw = (1 / m) * (X_train.T @ error)
            db = (1 / m) * np.sum(error, axis=0, keepdims=True)

            # Update
            self.w -= self.eta * dw
            self.b -= self.eta * db

            # Logging
            if self.verbose and epoch % 100 == 0:
                msg = f"[η={self.eta}] Epoch {epoch:04d} | Loss={loss:.6f}"
                if X_val is not None and y_val is not None:
                    val_acc = self.score(X_val, y_val)
                    msg += f" | Val Acc={val_acc:.4f}"
                print(msg)

            # Early stopping
            if loss < self.threshold:
                print(f"Converged at epoch {epoch} | Loss={loss:.6f}")
                break

        final_epoch = epoch if loss >= self.threshold else epoch + 1
        print(f"Training complete after {final_epoch} epochs | Final Loss: {loss:.6f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X = np.array(X, dtype=np.float64)
        z = X @ self.w + self.b
        return self._softmax(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy score."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_loss_history(self) -> np.ndarray:
        """Return training loss history."""
        return np.array(self.loss_history)

    # ===================================================================
    # Advanced Reporting
    # ===================================================================
    def predict_with_proba(
        self,
        X: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        show_samples: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with detailed probability output and optional true labels.

        Parameters
        ----------
        X : ndarray
        y_true : ndarray, optional
        show_samples : int

        Returns
        -------
        y_pred, y_proba
        """
        X = np.array(X, dtype=np.float64)
        y_proba = self.predict_proba(X)
        y_pred = self.predict(X)

        print("\n" + "="*80)
        print(" SOFTMAX REGRESSION - DỰ ĐOÁN CHI TIẾT")
        print("="*80)
        header = f"{'Index':<6} {'y_true':<8} {'y_pred':<8} {'Probabilities':<40} {'Correct?'}"
        print(header)
        print("-"*80)

        correct = 0
        n_samples = len(y_pred)

        for i in range(n_samples):
            if y_true is not None:
                y_t = y_true[i]
                correct_flag = "Correct" if y_t == y_pred[i] else "Wrong"
                if y_t == y_pred[i]:
                    correct += 1
            else:
                y_t = '?'
                correct_flag = ''

            prob_str = str(np.round(y_proba[i], 3)).replace('\n', '')
            if i < show_samples:
                print(f"{i:<6} {y_t:<8} {y_pred[i]:<8} {prob_str:<40} {correct_flag}")

        print("-"*80)
        if y_true is not None:
            acc = correct / n_samples
            print(f"Accuracy: {acc:.4f} ({correct}/{n_samples})")
        print(f"Learning rate: {self.eta} | Epochs: {len(self.loss_history)}")
        print("="*80 + "\n")

        return y_pred, y_proba

    # ===================================================================
    # Visualization
    # ===================================================================
    def plot_loss_curve(self, figsize=(8, 5), save_path: Optional[str] = None):
        """Plot training loss curve."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.loss_history, color='teal', linewidth=2)
        ax.set_title("Softmax Regression - Training Loss Curve", fontsize=14, pad=15)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss curve saved to {save_path}")
        plt.show()