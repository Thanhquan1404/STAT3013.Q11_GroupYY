import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score

class XGBoostClassifier:
    """
    XGBoost Classifier wrapper with sklearn-style interface.
    Supports both binary and multi-class classification.
    """

    def __init__(self, objective="multi:softmax", eta=0.05, max_depth=6,
                 subsample=0.8, colsample_bytree=0.8, num_boost_round=500,
                 early_stopping_rounds=None, verbose=True):
        """
        Initialize XGBoost classifier with hyperparameters.
        """
        self.objective = objective
        self.eta = eta
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

        self.model = None
        self.num_class = None
        self.evals_result = {}

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost model on training data.
        """
        unique_classes = np.unique(y_train)
        self.num_class = len(unique_classes)

        dtrain = xgb.DMatrix(X_train, label=y_train)

        params = {
            "objective": self.objective,
            "eta": self.eta,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "eval_metric": "mlogloss" if self.num_class > 2 else "logloss"
        }

        if self.num_class > 2:
            params["num_class"] = self.num_class

        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "validation"))

        if self.verbose:
            print(f"Training started | eta={self.eta}, max_depth={self.max_depth}, rounds={self.num_boost_round}")
            print(f"Classes: {self.num_class} | Objective: {self.objective}")

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            evals_result=self.evals_result,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=100 if self.verbose else False
        )

        if self.verbose:
            final_round = self.model.best_iteration if self.early_stopping_rounds else self.num_boost_round
            print(f"Training completed after {final_round} rounds.")
            if X_val is not None:
                val_acc = self.score(X_val, y_val)
                print(f"   Final Validation Accuracy: {val_acc:.4f}")

        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """
        Return predicted probabilities for each class.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")

        dtest = xgb.DMatrix(X)


        if "softmax" in self.objective:
            raw_predictions = self.model.predict(dtest, output_margin=True)

            if self.num_class > 2:
                exp_preds = np.exp(raw_predictions - np.max(raw_predictions, axis=1, keepdims=True))
                return exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
            else:
                return 1 / (1 + np.exp(-raw_predictions))
        else:
            return self.model.predict(dtest)

    # ------------------------------------------------------------------
    def predict(self, X):
        """
        Return class predictions.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")

        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest).astype(int)

    # ------------------------------------------------------------------
    def score(self, X, y):
        """
        Return accuracy score.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    # ------------------------------------------------------------------
    def get_evaluation_history(self):
        """
        Return evaluation metrics history during training.
        """
        return self.evals_result

    # ------------------------------------------------------------------
    def get_params(self):
        """Return model hyperparameters (sklearn style)."""
        return {
            "objective": self.objective,
            "eta": self.eta,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
            "verbose": self.verbose
        }

    def set_params(self, **params):
        """Set model hyperparameters (sklearn style)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    def feature_importance(self, importance_type='weight'):
        """
        Get feature importance scores.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")

        return self.model.get_score(importance_type=importance_type)

    # ------------------------------------------------------------------
    def predict_with_debug(self, X, y_true=None, show_samples=10):
        """
        Show detailed predictions with probabilities.
        Useful for debugging model performance.
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        print("\n" + "="*80)
        print("                 PREDICTION DETAILS")
        print("="*80)

        header = f"{'Index':<6} {'y_true':<8} {'y_pred':<8}"
        if y_proba.ndim > 1:
            for cls in range(y_proba.shape[1]):
                header += f"{'P('+str(cls)+')':<12}"
        else:
            header += f"{'Proba':<12}"
        header += f"{'Correct?'}"
        print(header)
        print("-"*80)

        correct = 0
        for i in range(min(len(y_pred), show_samples)):
            pred = int(y_pred[i])

            if y_true is not None:
                true = int(y_true[i])
                ok = "✔️" if pred == true else "✖️"
                if pred == true:
                    correct += 1
            else:
                true = "?"
                ok = ""

            row = f"{i:<6} {true:<8} {pred:<8}"

            if y_proba.ndim > 1:
                for prob in y_proba[i]:
                    row += f"{prob:<12.4f}"
            else:
                row += f"{y_proba[i]:<12.4f}"

            row += ok
            print(row)

        if y_true is not None:
            acc = accuracy_score(y_true, y_pred)
            print("-"*80)
            print(f"Overall Accuracy: {acc:.4f} ({int(acc * len(y_pred))}/{len(y_pred)})")
            print(f"Settings → eta={self.eta}, max_depth={self.max_depth}, rounds={self.num_boost_round}")
            print("\n" + "="*80)
            print("           CLASSIFICATION REPORT")
            print("="*80)
            print(classification_report(y_true, y_pred))

        print("="*80 + "\n")

        return y_pred, y_proba
