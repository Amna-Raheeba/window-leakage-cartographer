"""A deliberately simple predictor: k-nearest-neighbour regression.

kNN is the honest choice for a leakage audit. It has no training
dynamics to hide behind: a test window's prediction is literally an
average of its nearest train windows' labels, so when a near-copy of
the test window sits in the training set, the prediction is pulled to
that copy's label. The score inflation the cartographer measures is
therefore attributable to the leaked neighbours themselves, not to a
model class that might memorise in subtler ways.

Features are standardised with the training set's per-feature mean
and standard deviation.
"""

from .models import mean, std


class KNNRegressor:
    def __init__(self, k=3):
        self.k = k
        self.train_x = []
        self.train_y = []
        self.center = []
        self.scale = []

    def fit(self, features, labels):
        self.train_x = [tuple(f) for f in features]
        self.train_y = [float(y) for y in labels]
        n_feat = len(self.train_x[0]) if self.train_x else 0
        self.center = []
        self.scale = []
        for j in range(n_feat):
            column = [row[j] for row in self.train_x]
            self.center.append(mean(column))
            s = std(column)
            self.scale.append(s if s > 1e-12 else 1.0)
        return self

    def _distance(self, a, b):
        total = 0.0
        for j in range(len(a)):
            d = (a[j] - self.center[j]) / self.scale[j] - (
                b[j] - self.center[j]
            ) / self.scale[j]
            total += d * d
        return total

    def predict_one(self, features):
        if not self.train_x:
            return 0.0
        scored = []
        for idx, row in enumerate(self.train_x):
            scored.append((self._distance(features, row), idx))
        scored.sort()
        k = min(self.k, len(scored))
        return mean([self.train_y[idx] for _dist, idx in scored[:k]])

    def predict(self, features_list):
        return [self.predict_one(f) for f in features_list]
