import csv
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

def remove_carrots_from_smiles(input_smiles):
    new_smiles_list = [''] * len(input_smiles)

    for i, smiles in enumerate(input_smiles):
        smiles = smiles.replace('>', '.')
        if 'hv' in smiles:
            smiles = smiles.replace('hv', '[H][V]')
        if '..' in smiles:
            smiles = smiles.replace('..', '.[Nd].')
        new_smiles_list[i] = smiles.rstrip('.')

    return new_smiles_list

def read_csv_data(filename, input_column, target_columns=[]):
    data = ([], [])
    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in reader:
            data[0].append(row[input_column])
            target_values = [float(row[col]) for col in target_columns]
            data[1].append(target_values)
    return map(np.array, data)

def load_data_without_slices(filename, input_column, target_columns):
    data = read_csv_data(filename, input_column, target_columns)
    return (data[0], data[1])

def slice_tuple(data, indices):
    return tuple([v[indices] for v in data])

def prepare_train_test_sets(data_file, input_column, target_columns, num_train_samples):
    np.random.seed(42)
    print("Loading data...")
    all_data = load_data_without_slices(data_file, input_column, target_columns)

    num_data = len(all_data[0])
    shuffled_indices = np.random.permutation(num_data)
    train_indices = shuffled_indices[:num_train_samples]
    test_indices = shuffled_indices[num_train_samples:]

    return slice_tuple(all_data, train_indices), slice_tuple(all_data, test_indices)

def split_smiles_strings(input_smiles):
    smile1_list = [''] * len(input_smiles)
    smile2_list = [''] * len(input_smiles)

    for i, smiles in enumerate(input_smiles):
        try:
            smi1, smi2 = smiles.split('.')
            smile1_list[i] = smi1
            smile2_list[i] = smi2
        except ValueError:
            print('Error: No dot found in reaction string')

    return list(zip(smile1_list, smile2_list))

def split_smiles_triples(input_smiles):
    smile1_list = [''] * len(input_smiles)
    smile2_list = [''] * len(input_smiles)
    smile3_list = [''] * len(input_smiles)
    smile4_list = [''] * len(input_smiles)

    for i, smiles in enumerate(input_smiles):
        try:
            reactants, reagents, products = smiles.split('>')
            smi1, smi2 = reactants.split('.')
            smile1_list[i] = smi1
            smile2_list[i] = smi2

            if reagents == 'hv':
                reagents = reagents.replace('hv', '[H][V]')
            smile3_list[i] = reagents
            smile4_list[i] = products
        except ValueError:
            print('Error: Invalid reaction string format')
            print(smiles)
            break

    return (list(zip(smile1_list, smile2_list, smile3_list)),
            list(zip(smile1_list, smile2_list, smile4_list)),
            list(zip(smile1_list, smile4_list)))

def calculate_confusion_matrix(true_matrix, unnormalized_pred_matrix, num_outputs):
    csr_true_matrix = csr_matrix(true_matrix)
    conf_matrix = np.zeros((num_outputs, num_outputs))
    row_indices, col_indices = csr_true_matrix.nonzero()

    pred_matrix = np.exp(unnormalized_pred_matrix - logsumexp(unnormalized_pred_matrix, axis=1, keepdims=True))
    reaction_counts = np.ones(num_outputs) * 0.0001

    for i in range(len(row_indices)):
        conf_matrix[:, col_indices[i]] += csr_true_matrix[row_indices[i], col_indices[i]] * pred_matrix[row_indices[i], :]
        reaction_counts[col_indices[i]] += csr_true_matrix[row_indices[i], col_indices[i]]

    conf_matrix /= reaction_counts

    print('Test normalization of columns:')
    print(np.sum(conf_matrix, axis=0))

    return reaction_counts, conf_matrix

def get_normalized_predictions(unnormalized_pred_matrix):
    return np.exp(unnormalized_pred_matrix - logsumexp(unnormalized_pred_matrix, axis=1, keepdims=True))

def calculate_accuracy(predictions, targets):
    is_max_pred = [[val == max(row) for val in row] for row in predictions]
    is_max_target = [[val == max(row) for val in row] for row in targets]
    return float(sum([is_max_pred[i] == is_max_target[i] for i in range(len(predictions))])) / len(predictions)

def calculate_L1_error(predictions, targets, num_outputs):
    predictions = predictions - logsumexp(predictions, axis=1, keepdims=True)
    pred_probs = np.exp(predictions)
    return np.mean(np.linalg.norm(pred_probs - targets, axis=1))
