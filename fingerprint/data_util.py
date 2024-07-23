import csv
import numpy as np
import numpy.random as npr

from util import slicedict
from rdkit_utils import smile_to_fp

def is_valid_smiles(smile):
    try:
        smile_to_fp(smile, 1, 10)
        return True
    except Exception as e:
        print(f"Couldn't parse {smile}: {e}")
        return False

def contains_duplicates(array):
    return len(set(array)) != len(array)

def remove_duplicates_with_key(values, key_func):
    seen = set()
    unique_values = []
    for value in values:
        key = key_func(value)
        if key not in seen:
            unique_values.append(value)
            seen.add(key)
    return unique_values

def shuffle_data(data):
    data = {k: np.array(v) for k, v in data.items()}  # To array for fancy indexing
    N = len(next(iter(data.values())))
    rand_ix = np.arange(N)
    npr.RandomState(0).shuffle(rand_ix)
    return slicedict(data, rand_ix)

def load_csv_to_dict(filename, column_indices, column_names, column_types, has_header=False, **kwargs):
    data = {name: [] for name in column_names}
    with open(filename) as file:
        reader = csv.reader(file, **kwargs)
        for row_number, row in enumerate(reader):
            if has_header and row_number == 0:
                continue
            try:
                for name, index, col_type in zip(column_names, column_indices, column_types):
                    data[name].append(col_type(row[index]))
            except Exception as e:
                print(f"Couldn't parse row {row_number}: {e}")
    return data

def filter_with_predicate(predicate, test_values, return_values):
    return [x[1] for x in filter(lambda x: predicate(test_values[x[0]]), enumerate(return_values))]

def filter_dictionary(data, key, predicate):
    filtered_values = data[key]
    return {k: filter_with_predicate(predicate, filtered_values, v) for k, v in data.items()}

def is_valid_shape(data):
    column_names = data.keys()
    length = len(data[next(iter(column_names))])
    return all(len(data[name]) == length for name in column_names)

def save_dict_to_csv(data, filename):
    column_names = data.keys()
    with open(filename, 'w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='\'')
        writer.writerow(column_names)
        for row in zip(*(data[name] for name in column_names)):
            writer.writerow(row)
