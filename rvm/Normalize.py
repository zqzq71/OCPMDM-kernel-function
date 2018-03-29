import numpy as np

class FeatureScaler():
    def __init__(self,method):
        self.method = method
        self.params = {}

    def transform(self,X):
        if self.method == 'normalize':
            result = (X-self.params['means']) / self.params['std']
        elif self.method == 'minmax':
            result = (X - self.params['min_value']) / (self.params['max_value'] - self.params['min_value'])
        elif self.method == 'normalstd':
            result = (X - self.params['means']) / self.params['ava']
        else:
            return 'Unknow Method'
        return result

    # normalize: (x-mean) / variance
    # standard: (xmax-x)/(xmax-xmin)
    # normalstd : (x-mean) / ava
    def __call__(self,X):

        # 20170613 add convert X to np.float32
        X = np.asarray(X, dtype='float32')

        if self.method == 'normalize':
            means = np.mean(X,axis=0)
            std = np.std(X,axis=0)

            self.params['means'] = means
            self.params['std'] = std
        elif self.method == 'minmax':
            max_val = np.max(X,axis=0)
            min_val = np.min(X,axis=0)

            self.params['min_value'] = min_val
            self.params['max_value'] = max_val
        elif self.method == "normalstd":
            means = np.mean(X,axis=0)
            sample_num = X.shape[0]

            ava = np.sqrt(np.sum(np.square(X - means),axis=0) / (sample_num - 1))

            self.params['means'] = means
            self.params['ava'] = ava
        else:
            return 'Unknow Method'

        return self.transform(X)


class TargetMapping():
    def __init__(self):
        pass
    def transform(self,y):
        return (y-self.min) / (self.max - self.min)

    def recovery(self,y):
        return y * (self.max - self.min) + self.min

    def __call__(self,y):
        self.min = np.min(y)
        self.max = np.max(y)

        return self.transform(y)