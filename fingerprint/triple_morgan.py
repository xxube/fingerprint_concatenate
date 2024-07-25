
#Building reaction fingerprints 

import autograd.numpy as np
import autograd.numpy.random as npr
from util import memoize, WeightsParser
from rdkit_utils import convert_smiles_to_fingerprints
from convnet import build_convnet_fp_fun
from neural_net import build_fp_deep_net

def create_triple_conv_deep_net(conv_params, net_params, fp_l2_penalty=0.0):

    conv_fp_func, conv_parser = create_fixed_convnet_fp_func(**conv_params)
    return build_fingerprint_deep_net(net_params, conv_fp_func, conv_parser, fp_l2_penalty)

def create_fixed_convnet_fp_func(**kwargs):
    fp_fun, parser = build_convnet_fingerprint_fun(**kwargs)
    random_weights = npr.RandomState(0).randn(len(parser))

    def triple_fp_func(empty_weights, smiles_tuple):
        smiles1, smiles2, smiles3 = zip(*smiles_tuple)
        fp1 = fp_fun(random_weights, smiles1)
        fp2 = fp_fun(random_weights, smiles2)
        fp3 = fp_fun(random_weights, smiles3)
        return np.concatenate([fp1, fp2, fp3], axis=1)

    empty_parser = WeightsParser()
    return triple_fp_func, empty_parser

def create_triple_convnet_fp_func(**kwargs):
    fp_fun, parser = build_convnet_fingerprint_fun(**kwargs)

    def triple_fp_func(weights, smiles_tuple):
        smiles1, smiles2, smiles3 = zip(*smiles_tuple)
        fp1 = fp_fun(weights, smiles1)
        fp2 = fp_fun(weights, smiles2)
        fp3 = fp_fun(weights, smiles3)
        return np.concatenate([fp1, fp2, fp3], axis=1)

    return triple_fp_func, parser

def create_triple_morgan_deep_net(fp_length, fp_depth, net_params):
    empty_parser = WeightsParser()
    morgan_fp_func = create_triple_morgan_fp_func(fp_length, fp_depth)
    return build_fingerprint_deep_net(net_params, morgan_fp_func, empty_parser, 0)

def create_triple_morgan_fp_func(fp_length=512, fp_radius=4):
    def generate_fps_from_smiles(weights, smiles_tuple):
        # @pre: smiles_tuple is a list of three tuples 
        smiles1, smiles2, smiles3 = zip(*smiles_tuple)
        # Morgan fingerprints don't use weights.
        fp1 = generate_fps_from_smiles_tuple(tuple(smiles1))
        fp2 = generate_fps_from_smiles_tuple(tuple(smiles2))
        fp3 = generate_fps_from_smiles_tuple(tuple(smiles3))
        return np.concatenate([fp1, fp2, fp3], axis=1)

    @memoize # This wrapper function exists because tuples can be hashed, but arrays can't.
    def generate_fps_from_smiles_tuple(smiles_tuple):
        return smiles_to_fps(smiles_tuple, fp_length, fp_radius)

    return generate_fps_from_smiles
