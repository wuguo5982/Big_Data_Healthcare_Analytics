import math
class LogisticRegressionSGD:
    # Logistic regression with stochastic gradient descent
    def __init__(self, eta, mu, n_feature):
        # Initialization of model parameters
        self.eta = eta
        self.weight = [0.0] * n_feature
        self.mu = mu

    def fit(self, X, y):
        # Update model using a pair of training sample
        for u in X:
            gradient = (y - self.predict_prob(X)) * u[1]
            self.weight[u[0]] += self.eta * gradient - 2 * self.eta * self.mu * self.weight[u[0]]
        pass

    def predict(self, X):
        # Predict 0 or 1 given X and the current weights in the model
        return 1 if predict_prob(X) > 0.5 else 0

    def predict_prob(self, X):
        # Sigmoid function
        return 1.0 / (1.0 + math.exp(-math.fsum((self.weight[f]*v for f, v in X))))

