import autograd.numpy as np
from autograd.scipy.misc import logsumexp
import autograd.numpy.random as npr
from autograd import grad
from sklearn.base import BaseEstimator, RegressorMixin
from rdkit import Chem
from rdkit.Chem import AllChem
import inspect
from neuralfingerprint import build_double_morgan_deep_net, relu
from neuralfingerprint import build_double_conv_deep_net, adam
from neuralfingerprint import build_batched_grad, categorical_nll 
from neuralfingerprint import normalize_array

class ReactionEstimator(BaseEstimator, RegressorMixin):
    #classification 
    def __init__(self,
                 log_learn_rate=-5., 
                 log_init_scale=-6.,
                 fp_length=10,
                 other_param_dict=None): 
        if other_param_dict is None:
            other_param_dict = {'num_epochs': 100, 'batch_size': 200, 'normalize': 1,
                                'dropout': 0, 'fp_depth': 3, 'activation': relu, 'fp_type': 'morgan',
                                'h1_size': 100, 'conv_width': 20, 'num_outputs': 17, 'init_bias': 0.85}
        
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        for arg, val in values.items(): 
            setattr(self, arg, val)

    def get_training_params(self):
        nn_train_params = {'num_epochs': self.other_param_dict['num_epochs'],
                           'batch_size': self.other_param_dict['batch_size'],
                           'learn_rate': np.exp(self.log_learn_rate),
                           'param_scale': np.exp(self.log_init_scale)}
    
        vanilla_net_params = {'layer_sizes': [self.fp_length * 2, self.other_param_dict['h1_size']],
                              'normalize': self.other_param_dict['normalize'],
                              'activation_function': self.other_param_dict['activation'],
                              'nll_func': categorical_nll,
                              'num_outputs': self.other_param_dict['num_outputs']}
        
        if isinstance(self.other_param_dict['h1_size'], list):
            temp_layers = [self.other_param_dict['h1_size'][i] for i in range(len(self.other_param_dict['h1_size']))]
            vanilla_net_params['layer_sizes'] = [self.fp_length * 2] + temp_layers
        
        return nn_train_params, vanilla_net_params

    def train_network(self, pred_func, loss_func, num_weights, train_smiles, train_targets, train_params):
        print("Total number of weights in the network:", num_weights)
        npr.seed(0)
        init_weights = npr.randn(num_weights) * train_params['param_scale']
        init_weights[-1] = self.other_param_dict['init_bias']
    
        train_targets, undo_norm = normalize_array(train_targets)
        training_curve = []

        def callback(weights, iter):
            if iter % 20 == 0:
                print("max of weights", np.max(np.abs(weights)))
                train_preds = undo_norm(pred_func(weights, train_smiles))
                cur_loss = loss_func(weights, train_smiles, train_targets)
                training_curve.append(cur_loss)
                print("Iteration", iter, "loss", cur_loss)
    
        grad_func = grad(loss_func)
        grad_func_with_data = build_batched_grad(grad_func, train_params['batch_size'],
                                                 train_smiles, train_targets)
    
        num_iters = train_params['num_epochs'] * train_smiles.shape[0] // train_params['batch_size']
        trained_weights = adam(grad_func_with_data, init_weights, callback=callback,
                               num_iters=num_iters, step_size=train_params['learn_rate'])
    
        def prediction_function(new_smiles):
            return pred_func(trained_weights, new_smiles)

        return prediction_function, trained_weights, training_curve

    def fit(self, X, y):
        def run_morgan(X, y):
            nn_train_params, vanilla_net_params = self.get_training_params()
            print("Task params", nn_train_params, vanilla_net_params)

            fp_params = {'fp_length': self.fp_length, 'fp_radius': self.other_param_dict['fp_depth']}
            print("Morgan Fingerprints with neural net")
            
            loss_func, pred_func, net_parser = build_double_morgan_deep_net(fp_params['fp_length'], fp_params['fp_radius'], vanilla_net_params)
            num_weights = len(net_parser)
            predict_func, trained_weights, training_curve = self.train_network(pred_func, loss_func, num_weights, X, y, nn_train_params)
            return predict_func, trained_weights, training_curve

        def run_neural(X, y):
            nn_train_params, vanilla_net_params = self.get_training_params()
            print("Task params", nn_train_params, vanilla_net_params)
            
            fp_params = {'fp_length': self.fp_length, 'fp_radius': self.other_param_dict['fp_depth']}
            conv_layer_sizes = [self.other_param_dict['conv_width']] * fp_params['fp_radius']
            conv_arch_params = {'num_hidden_features': conv_layer_sizes, 'fp_length': fp_params['fp_length']}
            print("Neural Fingerprints with neural net")
            
            loss_func, pred_func, conv_parser = build_double_conv_deep_net(conv_arch_params, vanilla_net_params)
            num_weights = len(conv_parser)
            predict_func, trained_weights, training_curve = self.train_network(pred_func, loss_func, num_weights, X, y, nn_train_params)
            return predict_func, trained_weights, training_curve
        
        if self.other_param_dict['fp_type'] == 'morgan':
            self.predict_func, _, _ = run_morgan(X, y)
        elif self.other_param_dict['fp_type'] == 'neural':
            self.predict_func, _, _ = run_neural(X, y)

    def predict(self, X):
        return self.predict_func(X)

    def score(self, X, y):
        error = categorical_nll(self.predict(X), y, self.other_param_dict['num_outputs']) 
        return error
