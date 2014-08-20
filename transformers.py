import numpy as np
from sklearn.base import TransformerMixin


class ModelTransformer(TransformerMixin):
    """Wrap a classifier model so that it can be used in a pipeline"""
    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        print(X.shape)
        return self.model.predict_proba(X)

    def predict_proba(self, X, **transform_params):
        return self.transform(X, **transform_params)


class VarTransformer(TransformerMixin):
    """Compute the variance"""
    def transform(self, X, **transform_params):
        var = X.var(axis=1)
        return var.reshape((var.shape[0],1))

    def fit(self, X, y=None, **fit_params):
        return self


class MedianTransformer(TransformerMixin):
    """Compute the median"""
    def transform(self, X, **transform_params):
        median = np.median(X, axis=1)
        return median.reshape((median.shape[0],1))

    def fit(self, X, y=None, **fit_params):
        return self

class ChannelExtractor(TransformerMixin):
    """Extract a single channel for downstream processing"""
    def __init__(self, channel):
        self.channel = channel

    def transform(self, X, **transformer_params):
        return X[:,:,self.channel]

    def fit(self, X, y=None, **fit_params):
        return self


class FFTTransformer(TransformerMixin):
    """Convert to the frequency domain and then sum over bins
    TODO: Choose optimal bin_size and max_freq"""
    def transform(self, X, **transformer_params):
        fft = np.fft.rfft(X, axis=1)
        fft = np.abs(fft)
        fft = np.cumsum(fft, axis=1)
        bin_size = 10
        max_freq = 60
        return np.column_stack([fft[:,i] - fft[:,i-bin_size] 
            for i in range(bin_size, max_freq, bin_size)])

    def fit(self, X, y=None, **fit_params):
        return self



