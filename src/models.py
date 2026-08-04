import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class LinearRegressionCustom(BaseEstimator, RegressorMixin):
    """
    Кастомная реализация линейной регрессии.
    Поддерживает три метода обучения:
    - Analytical (нормальное уравнение)
    - Batch Gradient Descent
    - Stochastic Gradient Descent
    """
    
    def __init__(self, method='analytical', learning_rate=0.01, 
                 n_iterations=10000, random_state=None):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """Обучение модели."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        if self.method == 'analytical':
            self._fit_analytical(X, y)
        elif self.method == 'batch_gd':
            self._fit_batch_gd(X, y)
        elif self.method == 'sgd':
            self._fit_sgd(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def _fit_analytical(self, X, y):
        """Аналитическое решение (нормальное уравнение)."""
        X_bias = np.c_[np.ones(X.shape[0]), X]
        try:
            theta = np.linalg.inv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)
        
        self.bias = theta[0]
        self.weights = theta[1:]
    
    def _fit_batch_gd(self, X, y):
        """Batch Gradient Descent."""
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
        for iteration in range(self.n_iterations):
            y_pred = X.dot(self.weights) + self.bias
            error = y_pred - y
            dw = (2/n_samples) * X.T.dot(error)
            db = (2/n_samples) * np.sum(error)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def _fit_sgd(self, X, y):
        """Stochastic Gradient Descent."""
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
        for epoch in range(self.n_iterations):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(n_samples):
                xi = X_shuffled[i]
                yi = y_shuffled[i]
                
                y_pred = xi.dot(self.weights) + self.bias
                error = y_pred - yi
                
                dw = 2 * error * xi
                db = 2 * error
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
            # Защитная проверка от взрыва градиентов
            if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)):
                break
    
    def predict(self, X):
        """Предсказание."""
        return X.dot(self.weights) + self.bias


class RegularizedLinearRegression(BaseEstimator, RegressorMixin):
    """
    Регуляризованная линейная регрессия.
    Поддерживает: Ridge (L2), Lasso (L1), ElasticNet (L1 + L2).
    """
    
    def __init__(self, reg_type='ridge', alpha=1.0, learning_rate=0.01,
                 n_iterations=10000, random_state=None):
        self.reg_type = reg_type
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """Обучение модели."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        if self.reg_type == 'ridge':
            self._fit_ridge_analytical(X, y)
        else:
            self._fit_gradient_descent(X, y)
        
        return self
    
    def _fit_ridge_analytical(self, X, y):
        """Ridge регрессия (аналитическое решение)."""
        X_bias = np.c_[np.ones(X.shape[0]), X]
        n_features = X_bias.shape[1]
        
        I = np.eye(n_features)
        I[0, 0] = 0  # Не штрафуем bias
        
        try:
            theta = np.linalg.solve(X_bias.T.dot(X_bias) + self.alpha * I, 
                                   X_bias.T.dot(y))
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(X_bias.T.dot(X_bias) + self.alpha * I).dot(X_bias.T).dot(y)
        
        self.bias = theta[0]
        self.weights = theta[1:]
    
    def _fit_gradient_descent(self, X, y):
        """Градиентный спуск для Lasso и ElasticNet."""
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
        lr = self.learning_rate
        
        for iteration in range(self.n_iterations):
            y_pred = X.dot(self.weights) + self.bias
            error = y_pred - y
            
            dw = (2/n_samples) * X.T.dot(error)
            db = (2/n_samples) * np.sum(error)
            
            if self.reg_type == 'lasso':
                dw += self.alpha * np.sign(self.weights)
            elif self.reg_type == 'elastic':
                dw += self.alpha * np.sign(self.weights) + self.alpha * self.weights
            
            self.weights -= lr * dw
            self.bias -= lr * db
    
    def predict(self, X):
        """Предсказание."""
        return X.dot(self.weights) + self.bias