import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from util import memoize, WeightsParser
from rdkit_utils import convert_smiles_to_fingerprints

def create_mean_predictor(loss_func):
    parser = WeightsParser()
    parser.add_weights('mean', (1,))
    def loss_func(weights, smiles, targets):
        mean = parser.get(weights, 'mean')
        return loss_func(np.full(targets.shape, mean), targets)
    def predict_func(weights, smiles):
        mean = parser.get(weights, 'mean')
        return np.full((len(smiles),), mean)
    return loss_func, predict_func, parser

def create_morgan_fp_func(fp_length=512, fp_radius=4):
    def generate_fps_from_smiles(weights, smiles):
        return generate_fps_from_smiles_tuple(tuple(smiles))

    @memoize
    def generate_fps_from_smiles_tuple(smiles_tuple):
        return smiles_to_fps(smiles_tuple, fp_length, fp_radius)

    return generate_fps_from_smiles

def create_morgan_deep_net(fp_length, fp_depth, net_params):
    empty_parser = WeightsParser()
    morgan_fp_func = create_morgan_fp_func(fp_length, fp_depth)
    return build_fp_deep_net(net_params, morgan_fp_func, empty_parser, 0)

def batch_norm(activations):
    mbmean = np.mean(activations, axis=0, keepdims=True)
    return (activations - mbmean) / (np.std(activations, axis=0, keepdims=True) + 1)

def relu_activation(X):
    "Rectified linear activation function."
    return X * (X > 0)

def sigmoid_activation(x):
    return 0.5 * (np.tanh(x) + 1)

def mse(predictions, targets):
    return np.mean((predictions - targets) ** 2, axis=0)

def categorical_cross_entropy(predictions, targets, num_outputs):
    predictions = predictions - logsumexp(predictions, axis=1, keepdims=True)
    if not np.shape(predictions) == np.shape(targets):
        raise Exception("mismatch predictions {}, targets {}".format(str(np.shape(predictions)[0]), str(np.shape(targets)[0])))

    nll = 0.0
    for ex_id in range(np.shape(predictions)[0]):
        for outcome_id in range(num_outputs):
            if targets[ex_id, outcome_id] > 0.01:
                nll += targets[ex_id, outcome_id] * predictions[ex_id, outcome_id]
    return -nll / float(np.shape(targets)[0])

def binary_cross_entropy(predictions, targets):
    pred_probs = sigmoid_activation(predictions)
    label_probabilities = pred_probs * targets + (1 - pred_probs) * (1 - targets)
    return -np.mean(np.log(label_probabilities))

def build_neural_net(layer_sizes, normalize, L2_reg=0.0, L1_reg=0.0, activation_function=relu_activation,
                     nll_func=mse, num_outputs=1):
    layer_sizes = layer_sizes + [num_outputs]
    parser = WeightsParser()
    for i, shape in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        parser.add_weights(('weights', i), shape)
        parser.add_weights(('biases', i), (1, shape[1]))

    def predict(W_vect, X):
        cur_units = X
        for layer in range(len(layer_sizes) - 1):
            cur_W = parser.get(W_vect, ('weights', layer))
            cur_B = parser.get(W_vect, ('biases', layer))
            cur_units = np.dot(cur_units, cur_W) + cur_B
            if layer < len(layer_sizes) - 2:
                if normalize:
                    cur_units = batch_norm(cur_units)
                cur_units = activation_function(cur_units)
        return cur_units

    def loss_func(w, X, targets):
        assert len(w) > 0
        log_prior = -L2_reg * np.dot(w, w) / len(w) - L1_reg * np.mean(np.abs(w))
        preds = predict(w, X)
        if nll_func == categorical_cross_entropy:
            return nll_func(preds, targets, num_outputs) - log_prior
        return nll_func(preds, targets) - log_prior

    return loss_func, predict, parser

def build_fp_deep_net(net_params, fingerprint_func, fp_parser, fp_l2_penalty):
    net_loss_func, net_pred_func, net_parser = build_neural_net(**net_params)
    combined_parser = WeightsParser()
    combined_parser.add_weights('fingerprint weights', (len(fp_parser),))
    combined_parser.add_weights('net weights', (len(net_parser),))

    def unpack_weights(weights):
        fingerprint_weights = combined_parser.get(weights, 'fingerprint weights')
        net_weights = combined_parser.get(weights, 'net weights')
        return fingerprint_weights, net_weights

    def loss_func(weights, smiles, targets):
        fingerprint_weights, net_weights = unpack_weights(weights)
        fingerprints = fingerprint_func(fingerprint_weights, smiles)
        net_loss = net_loss_func(net_weights, fingerprints, targets)
        if len(fingerprint_weights) > 0 and fp_l2_penalty > 0:
            return net_loss + fp_l2_penalty * np.mean(fingerprint_weights ** 2)
        else:
            return net_loss

    def predict_func(weights, smiles):
        fingerprint_weights, net_weights = unpack_weights(weights)
        fingerprints = fingerprint_func(fingerprint_weights, smiles)
        return net_pred_func(net_weights, fingerprints)

    return loss_func, predict_func, combined_parser
