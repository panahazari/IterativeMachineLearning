import numpy as np

class LogisticRegressionDG_mp:
    def __init__(self, learning_rate=0.08, num_iterations=500):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None
        self.losses = []

    def add_bias_column(self, data):
        bias = np.ones((data.shape[0], 1))
        return np.hstack((data, bias))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, train_data, train_labels, test_data, test_labels):
        train_data = self.add_bias_column(train_data)
        test_data = self.add_bias_column(test_data)
        
        self.theta = np.random.randn(train_data.shape[1], 1)
        
        for k in range(self.num_iterations):
            predictions = self.sigmoid(np.dot(train_data, self.theta))
            predictions = np.clip(predictions, 1e-5, 1 - 1e-5)

            gradient = np.dot(train_data.T, (predictions - train_labels[:, np.newaxis])) / len(train_labels)
            self.theta -= self.learning_rate * gradient

            loss = -np.mean(train_labels[:, np.newaxis] * np.log(predictions) + (1 - train_labels[:, np.newaxis]) * np.log(1 - predictions))
            self.losses.append(loss)

        self.train_accuracy = np.mean((self.sigmoid(np.dot(train_data, self.theta)) >= 0.5).flatten() == train_labels)
        self.test_accuracy = np.mean((self.sigmoid(np.dot(test_data, self.theta)) >= 0.5).flatten() == test_labels)

    def predict(self, sample):
        sample = np.append(sample, 1)  # Add bias before prediction
        return (self.sigmoid(np.dot(sample, self.theta)) >= 0.5).astype(int)

    def get_train_accuracy(self):
        return self.train_accuracy

    def get_test_accuracy(self):
        return self.test_accuracy

    def get_losses(self):
        return self.losses

    def get_model(self):
        return self.theta

    def get_model_without_bias(self):
        return self.theta[:-1]
