import os
import csv
import numpy as np
import itertools as it

def get_env_variable(varname):
    if varname in os.environ:
        return os.environ[varname]
    else:
        raise Exception(f"{varname} environment variable not set")

def get_output_directory():
    return os.path.expanduser(get_env_variable("OUTPUT_DIR"))

def get_data_directory():
    return os.path.expanduser(get_env_variable("DATA_DIR"))

def get_output_filepath(relative_path):
    return os.path.join(get_output_directory(), relative_path)

def get_data_filepath(relative_path):
    return os.path.join(get_data_directory(), relative_path)

def read_csv_file(filename, num_rows, input_col, target_col):
    data = ([], [])
    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in it.islice(reader, num_rows):
            data[0].append(row[input_col])
            data[1].append(float(row[target_col]))
    return map(np.array, data)

def load_dataset(filename, sizes, input_col, target_col):
    slices = []
    start = 0
    for size in sizes:
        stop = start + size
        slices.append(slice(start, stop))
        start = stop
    return load_data_by_slices(filename, slices, input_col, target_col)

def load_data_by_slices(filename, slices, input_col, target_col):
    stops = [s.stop for s in slices]
    if not all(stops):
        raise Exception("Slices can't be open-ended")

    data = read_csv_file(filename, max(stops), input_col, target_col)
    return [(data[0][s], data[1][s]) for s in slices]

def concatenate_lists(lists):
    return list(it.chain(*lists))

def load_data_from_slices(filename, slice_lists, input_col, target_col):
    stops = [s.stop for s in concatenate_lists(slice_lists)]
    if not all(stops):
        raise Exception("Slices can't be open-ended")

    data = read_csv_file(filename, max(stops), input_col, target_col)

    return [(np.concatenate([data[0][s] for s in slices], axis=0),
             np.concatenate([data[1][s] for s in slices], axis=0))
            for slices in slice_lists]
