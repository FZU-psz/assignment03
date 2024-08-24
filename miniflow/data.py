import numpy as np

class Dataset:
    def __init__(self, name, data, labels):
        self.name = name
        self.data = data
        self.labels = labels
        self.index_in_epoch = 0
        self.num_examples = data.shape[0]
        self.epochs_completed = 0

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size

        if self.index_in_epoch > self.num_examples:
            self.epochs_completed += 1
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.data = self.data[perm]
            self.labels = self.labels[perm]
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        
        end = self.index_in_epoch
        return self.data[start:end], self.labels[start:end]

def load_mnist_data(data_path):
    import os
    import gzip
    import numpy as np

    def load_data(filename, label=False):
        with gzip.open(filename) as gz:
            data = np.frombuffer(gz.read(), dtype=np.uint8)
            data = data[16:].reshape(-1, 28*28).astype(np.float32)
        return data

    def load_labels(filename):
        with gzip.open(filename) as gz:
            labels = np.frombuffer(gz.read(), dtype=np.uint8, offset=8)
        return labels

    data = load_data(os.path.join(data_path, 'train-images-idx3-ubyte.gz'))
    labels = load_labels(os.path.join(data_path, 'train-labels-idx1-ubyte.gz'))
    return Dataset('train', data, labels)