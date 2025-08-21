# Copyright (C) 2022 Riccardo Simionato, University of Oslo
# Inquiries: riccardo.simionato.vib@gmail.com.com
#
# This code is free software: you can redistribute it and/or modify it under the terms
# of the GNU Lesser General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Less General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this code.
# If not, see <http://www.gnu.org/licenses/>.
#
# If you use this code or any part of it in any program or publication, please acknowledge
# its authors by adding a reference to this publication:
#
# R. Simionato, 2022, "Deep Learning Conditioned Modeling of Optical Compression" in proceedings of the 22th Digital Audio Effect Conference, Vienna, Austria.



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