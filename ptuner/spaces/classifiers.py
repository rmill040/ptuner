from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from math import log
import numpy as np
from typing import Any, Dict

# Custom imports
from ..base._sampler import BaseSampler

__all__ = [
    'MLPClassifierSampler',
    'XGBClassifierSampler'
]

class MLPClassifierSampler(BaseSampler):
    """ADD
    
    Parameters
    ----------
    """
    def __init__(
        self, 
        dynamic_update=True,
        early_stopping=True, 
        n_hidden_layers=1, 
        max_neurons=512,
        max_epochs=10,
        seed=None
        ):
        self.n_hidden_layers  = n_hidden_layers
        self.max_neurons      = max_neurons
        self.max_epochs       = max_epochs
        self.early_stopping   = early_stopping
        super().__init__(dynamic_update=dynamic_update, seed=seed)
    

    def __str__(self) -> str:
        return "MLPClassifierSampler"


    def _starting_space(self) -> Dict[str, Any]:
        """ADD HERE"""
        params                    = {}
        params['n_hidden_layers'] = self.n_hidden_layers

        # Add all hidden layer units
        for i in range(1, self.n_hidden_layers+1):
            params['n_hidden%s' % i] = hp.quniform('mlp_h%s' % i, 1, self.max_neurons, 1)

        params.update({
            'learning_rate' : hp.loguniform('mlp_lr', log(1e-4), log(1)),
            'dropout'       : hp.uniform('mlp_do', 0, 1),
            'epochs'        : hp.quniform('mlp_e', 1, self.max_epochs, 1),
            'batch_size'    : hp.quniform('mlp_bs', 2, 512, 1),
            'batch_norm'    : hp.pchoice('mlp_bn', [(0.5, 'no'), (0.5, 'yes')]),
            'optimizer'     : hp.pchoice('mlp_opt', [(0.5, 'adam'), (0.5, 'sgd')]),
            'reg_l1'        : hp.loguniform('mlp_l1', log(1e-4), log(1)),
            'reg_l2'        : hp.loguniform('mlp_l2', log(1e-4), log(1))
        })

        return params


    def sample_space(self) -> Dict[str, Any]:
        """ADD DESCRIPTION"""
        hypers = {}
        for param, dist in self.space.items():
            value = sample(dist)
            if 'hidden' in param or param in ['epochs', 'batch_size']:
                hypers[param] = int(value)
            else:
                hypers[param] = value
        
        hypers['n_hidden_layers'] = self.n_hidden_layers

        return hypers


    def update_space(self, data) -> None:
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """        
        if not self.dynamic_update: return
    
        # Update search distributions for hyperparameters
        for param in self.space.keys():
            
            if param in ['batch_norm', 'optimizer']:
                pchoice = data[param].value_counts(True).sort_index().to_dict()
                pchoice = [(value, key) for key, value in pchoice.items()]
                
                # Update label
                label             = 'bn' if param == 'batch_norm' else 'opt'
                label             = 'mlp_' + label
                self.space[param] = hp.pchoice(label, pchoice)
            else:
                # Min and max values of current parameter
                min_value, max_value = data[param].min(), data[param].max()
                
                if 'n_hidden' in param:
                    label             = 'mlp_h' + param[-1]
                    self.space[param] = hp.quniform(label, min_value, max_value, 1)

                elif param == 'learning_rate':
                    self.space[param] = hp.loguniform('mlp_lr', log(min_value), log(max_value))

                elif param == 'dropout':
                    self.space[param] = hp.uniform('mlp_do', min_value, max_value)
                
                # Do not update epochs if early stopping enabled
                elif param == 'epochs':
                    if self.early_stopping: continue
                    self.space[param] = hp.quniform('mlp_e', min_value, max_value, 1)

                elif param == 'batch_size':
                    self.space[param] = hp.quniform('mlp_bs', min_value, max_value, 1)

                elif param == 'reg_l1':
                    self.space[param] = hp.loguniform('mlp_l1', log(min_value), log(max_value))

                elif param == 'reg_l2':
                    self.space[param] = hp.loguniform('mlp_l2', log(min_value), log(max_value))

                else:
                    raise ValueError("%s param not found in sample space" % param)


class XGBClassifierSampler(BaseSampler):
    """ADD
    
    Parameters
    ----------
    """
    def __init__(
        self, 
        dynamic_update=True, 
        early_stopping=True, 
        seed=1718
        ) -> None:
        self.early_stopping = early_stopping
        super().__init__(dynamic_update=dynamic_update, seed=seed)


    def __str__(self) -> str:
        """ADD HERE.
        """
        return "XGBClassifierSampler"


    def _starting_space(self) -> Dict[str, Any]:
        """ADD HERE.
        """
        return {
            'n_estimators'      : hp.quniform('xgb_ne', 10, 2000, 1),
            'learning_rate'     : hp.loguniform('xgb_lr', log(1e-3), log(1)),
            'max_depth'         : hp.quniform('xgb_md', 1, 12, 1),
            'min_child_weight'  : hp.quniform('xgb_mcw', 1, 20, 1),
            'subsample'         : hp.uniform('xgb_ss', 0.20, 1.0),
            'colsample_bytree'  : hp.uniform('xgb_cbt', 0.20, 1.0),
            'colsample_bylevel' : hp.uniform('xgb_cbl', 0.20, 1.0),
            'gamma'             : hp.uniform('xgb_g', 0, 1),
            'reg_alpha'         : hp.loguniform('xgb_ra', log(1e-10), log(1e-1)),
            'reg_lambda'        : hp.loguniform('xgb_rl', log(1e-10), log(1e-1)),
            'base_score'        : hp.uniform('xgb_bs', 0.01, 0.99),
            'max_delta_step'    : hp.quniform('xgb_mds', 0, 3, 1),
            'scale_pos_weight'  : hp.uniform('xgb_spw', 0.1, 10)
            }


    def sample_space(self) -> Dict[str, Any]:
        """ADD DESCRIPTION.
        
        Parameters
        ----------
        None

        Returns
        -------
        """
        hypers = {}
        for param, dist in self.space.items():
            hypers[param] = int(sample(dist)) \
                if param in ['n_estimators', 'max_delta_step', 'max_depth', 'min_child_weight'] \
                    else sample(dist)
        
        # Add seed
        hypers['seed'] = self.seed

        return hypers


    def update_space(self, data: Any) -> None:
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """    
        if not self.dynamic_update: return

        # Update search distributions for hyperparameters
        for param in self.space.keys():
            
            # Min and max values of current parameter
            min_value, max_value = data[param].min(), data[param].max()
            
            # Do not update n_estimators if early_stopping enabled
            if param == 'n_estimators':
                if self.early_stopping: continue
                self.space[param] = hp.quniform('xgb_ne', min_value, max_value, 1)
            
            if param == 'learning_rate':
                self.space[param] = hp.loguniform('xgb_lr', log(min_value), log(max_value))
            
            elif param == 'max_depth': 
                self.space[param] = hp.quniform('xgb_md', min_value, max_value, 1)
            
            elif param == 'min_child_weight': 
                self.space[param] = hp.quniform('xgb_mcw', min_value, max_value, 1)
            
            elif param == 'subsample':
                self.space[param] = hp.uniform('xgb_ss', min_value, max_value)
            
            elif param == 'colsample_bytree':
                self.space[param] = hp.uniform('xgb_cbt', min_value, max_value)

            elif param == 'colsample_bylevel': 
                self.space[param] = hp.uniform('xgb_cbl', min_value, max_value)
            
            elif param == 'gamma':
                self.space[param] = hp.uniform('xgb_g', min_value, max_value)
            
            elif param == 'reg_alpha':
                self.space[param] = hp.loguniform('xgb_ra', log(min_value), log(max_value))

            elif param == 'reg_lambda':
                self.space[param] = hp.loguniform('xgb_rl', log(min_value), log(max_value))
            
            elif param == 'base_score':
                self.space[param] = hp.uniform('xgb_bs', min_value, max_value)
            
            elif param == 'max_delta_step':
                self.space[param] = hp.quniform('xgb_mds', min_value, max_value, 1)
            
            elif param == 'scale_pos_weight':
                self.space[param] =  hp.uniform('xgb_spw', min_value, max_value)
            
            else:
                raise ValueError("%s param not found in sample space" % param)