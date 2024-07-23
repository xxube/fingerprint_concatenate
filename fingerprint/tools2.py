import autograd.numpy as np
import autograd.numpy.random as npr
import sys, signal, pickle
from contextlib import contextmanager
from time import time
from functools import partial
from collections import OrderedDict

class Memoize:
    def __init__(self, function):
        self.function = function
        self.cache = {}

    def __call__(self, *args):
        if args in self.cache:
            return self.cache[args]
        else:
            result = self.function(*args)
            self.cache[args] = result
            return result

    def __get__(self, obj, objtype):
        return partial(self.__call__, obj)

class WeightsManager:
    def __init__(self):
        self.indices_and_shapes = OrderedDict()
        self.total_length = 0

    def add_weights(self, name, shape):
        start = self.total_length
        self.total_length += np.prod(shape)
        self.indices_and_shapes[name] = (slice(start, self.total_length), shape)

    def get_weights(self, vector, name):
        indices, shape = self.indices_and_shapes[name]
        return np.reshape(vector[indices], shape)

    def set_weights(self, vector, name, value):
        indices, _ = self.indices_and_shapes[name]
        vector[indices] = np.ravel(value)

    def __len__(self):
        return self.total_length

def encode_one_hot(value, allowable_set):
    if value not in allowable_set:
        raise Exception(f"Input {value} not in allowable set {allowable_set}")
    return list(map(lambda s: value == s, allowable_set))

def encode_one_hot_unknown(value, allowable_set):
    if value not in allowable_set:
        value = allowable_set[-1]
    return list(map(lambda s: value == s, allowable_set))

def normalize_data(array):
    mean, std = np.mean(array), np.std(array)
    normalized_array = (array - mean) / std
    def restore_function(data):
        return data * std + mean
    return normalized_array, restore_function

def root_mean_square_error(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))

def slice_dictionary(dictionary, indices):
    return {key: value[indices] for key, value in dictionary.items()}

def gather_test_losses(num_folds):
    results = {}
    for net_type in ['conv', 'morgan']:
        results[net_type] = []
        for fold_index in range(num_folds):
            filename = f"Final_test_loss_{fold_index}_{net_type}.pkl.save"
            try:
                with open(filename, 'rb') as file:
                    results[net_type].append(pickle.load(file))
            except IOError:
                print(f"Couldn't find file {filename}")

    print("Results are:")
    print(results)
    print("Means:")
    print({k: np.mean(v) for k, v in results.items()})
    print("Std errors:")
    print({k: np.std(v) / np.sqrt(len(v) - 1) for k, v in results.items()})

def save_loss(loss, experiment_index, net_type):
    filename = f"Final_test_loss_{experiment_index}_{net_type}.pkl.save"
    with open(filename, 'wb') as file:
        pickle.dump(float(loss), file)

def split_data_for_n_fold(N_folds, fold_index, N_data):
    fold_index = fold_index % N_folds
    fold_size = N_data // N_folds
    test_start = fold_size * fold_index
    test_end = fold_size * (fold_index + 1)
    test_indices = list(range(test_start, test_end))
    train_indices = list(range(0, test_start)) + list(range(test_end, N_data))
    return train_indices, test_indices

def apply_dropout(weights, fraction, random_state):
    mask = random_state.rand(len(weights)) > fraction
    return weights * mask / (1 - fraction)

def get_minibatch_indices(i, num_datapoints, batch_size):
    num_minibatches = num_datapoints // batch_size + ((num_datapoints % batch_size) > 0)
    i = i % num_minibatches
    start = i * batch_size
    stop = start + batch_size
    return slice(start, stop)

def create_batched_gradient(gradient, batch_size, inputs, targets):
    def batched_gradient(weights, i):
        current_indices = get_minibatch_indices(i, len(targets), batch_size)
        return gradient(weights, inputs[current_indices], targets[current_indices])
    return batched_gradient

def add_dropout_to_gradient(gradient, dropout_fraction, seed=0):
    assert(dropout_fraction < 1.0)
    def dropout_gradient(weights, i):
        mask = npr.RandomState(seed * 10**6 + i).rand(len(weights)) > dropout_fraction
        masked_weights = weights * mask / (1 - dropout_fraction)
        return gradient(masked_weights, i)
    return dropout_gradient

def measure_time():
    print("--- Start clock ---")
    start_time = time()
    yield
    elapsed_time = time() - start_time
    print(f"--- Stop clock: {elapsed_time} seconds elapsed ---")

def execute_with_error_handling(run_function, error_function):
    def signal_handler(signal, frame):
        error_function()
        sys.exit(0)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        result = run_function()
    except:
        error_function()
        raise
    return result
