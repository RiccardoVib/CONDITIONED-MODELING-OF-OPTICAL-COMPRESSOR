import numpy as np
import copy

class my_scaler():
    def __init__(self, feature_range=(0, 1)):
        super(my_scaler, self).__init__()
        self.max = feature_range[-1]
        self.min = feature_range[0]

    def fit(self, data):
        self.min_data = np.min(data)
        self.max_data = np.max(data)
        self.dtype = data.dtype

    def transform(self, data):
        X = copy.deepcopy(data)
        X_std = (X - self.min_data) / (self.max_data - self.min_data)
        return X_std * (self.max - self.min) + self.min

    def inverse_transform(self, data):
        X_scaled = copy.deepcopy(data)
        X_std = (X_scaled - self.min) / (self.max - self.min)
        X = X_std * (self.max_data - self.min_data) + self.min_data
        return X.astype(self.dtype)