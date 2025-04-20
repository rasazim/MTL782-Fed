import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight


class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    

    def __init__(self, C=1.0, fit_intercept=True, loss=None, grad=None,
                 class_weight='balanced', random_state=None, prox_mu = 0.1, max_iter=1):
        self.C = C
        self.fit_intercept = fit_intercept
        self.loss = loss
        self.grad = grad
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.prox_mu = prox_mu

    def fit(self, X, y, global_w, sample_weight=None):

        self.global_w=global_w
        X, y = check_X_y(X, y, accept_sparse=False)  
        self.classes_ = unique_labels(y)

        if len(self.classes_) != 2:
            raise ValueError("Only binary classification is supported.")

        self.le_ = LabelEncoder().fit(y)  
        y = self.le_.transform(y) 

        n_samples, n_features = X.shape

        if self.fit_intercept:
            X = np.c_[np.ones(n_samples), X]  

        if self.class_weight == 'balanced':
            class_weights = class_weight.compute_class_weight('balanced', classes=self.classes_, y=y)
            
            sample_weight = (
                np.ones(n_samples) if sample_weight is None else sample_weight
            )
            adjusted_sample_weight = np.array([class_weights[i] for i in y]) * sample_weight
        elif isinstance(self.class_weight, dict):
            if set(self.class_weight.keys()) != set(self.classes_):
                raise ValueError(
                    "Class labels in `class_weight` do not match the labels"
                    "found in the data: %r != %r" % (
                        set(self.class_weight.keys()), set(self.classes_)))
            sample_weight = (
                np.ones(n_samples) if sample_weight is None else sample_weight
            )

            adjusted_sample_weight = np.array([self.class_weight[c] for c in y]) * sample_weight
        else:
            adjusted_sample_weight = sample_weight 

        initial_w = np.concatenate((self.intercept_, self.coef_.flatten()))

        def objective(w):
            y_pred = self._sigmoid(X.dot(w))
            loss_val = self.loss(y, y_pred, adjusted_sample_weight) if self.loss else self._logistic_loss(y, y_pred, adjusted_sample_weight)
            reg_term = 0.5 * (1 / self.C) * np.sum(w[self.fit_intercept:] ** 2)
            return loss_val + reg_term

        def gradient(w):
            y_pred = self._sigmoid(X.dot(w))
            grad_loss = self.grad(y, y_pred, adjusted_sample_weight, X) if self.grad else self._logistic_grad(y, y_pred, adjusted_sample_weight, X)
            grad_reg = (1 / self.C) * w 
            grad_reg[0] = 0 
            proximal_gradient = self.prox_mu * (w - global_w)
            return grad_loss + grad_reg - proximal_gradient

        
        if self.loss is not None and self.grad is None:
            result = minimize(objective, initial_w, method='L-BFGS-B', jac='2-point', options={'maxiter':self.max_iter})
        else:
            result = minimize(objective, initial_w, method='L-BFGS-B', jac=gradient, options={'maxiter':self.max_iter})
        


        self.coef_ = np.reshape(result.x[self.fit_intercept:],(1,-1))
        self.intercept_ = np.array([result.x[0]]) if self.fit_intercept else np.zeros(1)

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)  
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        probabilities = self._sigmoid(X.dot(np.concatenate((self.intercept_, self.coef_.flatten()))))
        return np.column_stack([1 - probabilities, probabilities]) 

    def predict(self, X):
        
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    def _sigmoid(self, z):
        
        return 1 / (1 + np.exp(-z))

    def _logistic_loss(self, y_true, y_pred, weights):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        if weights is None:
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            return -np.mean(weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    def _logistic_grad(self, y_true, y_pred, weights, X):
      
        if weights is None:
            return np.dot(X.T, (y_pred - y_true)) / y_true.shape[0]
        else:
            return np.dot(X.T, weights * (y_pred - y_true)) / y_true.shape[0]
