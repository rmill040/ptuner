from math import log
import pandas as pd
import pytest
import re

# Package imports
from ptuner.spaces import MLPClassifierSampler, XGBClassifierSampler
from ptuner.utils import helper

# Constants
n_hidden_layers = 2
max_neurons     = 512
max_epochs      = 10


def test_mlp_classifier_sampler():
    """Test MLPClassifierSampler.
    """
    # Define sampler
    sampler = MLPClassifierSampler(
        dynamic_update=True,
        early_stopping=True,
        n_hidden_layers=n_hidden_layers,
        max_neurons=max_neurons,
        max_epochs=max_epochs,
        seed=None
    )

    # Check 1. See if sampler is initialized with correct number of hidden layers
    assert(sampler.space['n_hidden_layers'] == n_hidden_layers), \
        "sampler initialized with incorrect number of hidden layers"

    # Check 2. See if correct number of hidden unit parameters are defined
    names      = filter(lambda x: 'n_hidden' in x and 'layers' not in x, sampler.space.keys())
    cols2check = ['n_hidden%s' % i for i in range(1, n_hidden_layers+1)]
    assert(set(names) == set(cols2check)), \
        "number of hidden unit parameters not defined correctly, check starting space"

    # Check 3. Make sure sampling changes values within bounds defined by user
    df_configs = pd.DataFrame([sampler.sample_space() for _ in range(20)])
    for column in df_configs:
        # Remember this hyperparameter should stay fixed
        if column == 'n_hidden_layers': 
            assert(df_configs[column].unique()[0] == n_hidden_layers), \
                "n_hidden_layers should not change during sampling"
            continue
        
        # Other hyperparameters should change across samples
        assert(df_configs[column].unique().shape[0] > 1), \
            "sampling generated constant values for %s, check sampling scheme" % column

    # Check 4. Make sure dynamic updating is working based on made up data
    df_configs = df_configs.sort_values(by="dropout", ascending=True)
    df_hof     = df_configs.copy(deep=True).iloc[:5]

    # Force minimum number of epochs to be greater than 1 for later assertion check
    df_hof.loc[df_hof['epochs'] == 1, 'epochs'] = 2

    # Now update space based on 'hof' candidates
    sampler.update_space(df_hof)
    
    for column in df_hof:
        # These are categorical distributions so skip
        if column in ['batch_norm', 'optimizer']: continue
        
        actual_min            = df_hof[column].min()
        actual_max            = df_hof[column].max()
        param_type, hp_bounds = helper.parse_hyperopt_param(str(sampler.space[column]))

        # For necessary hyperparameters, make sure scale is correct
        if param_type == 'loguniform':
            actual_min = log(actual_min)
            actual_max = log(actual_max)
        
        # Since early_stopping is enabled, this parameter should not be dynamically
        # updated => the range should still be [1, max_epochs]
        if column == 'epochs':
            assert(actual_min > 1), \
                "epochs should not be updated with early stopping enabled"
            continue

        assert(actual_min == hp_bounds[0]), \
            "lower bound did not update correctly for %s during space update" % column
        
        assert(actual_max == hp_bounds[1]), \
            "upper bound did not update correctly for %s during space update" % column
        


def test_xgb_classifier_sampler():
    """Tests XGBClassifierSampler.
    """
    # Define sampler
    sampler = XGBClassifierSampler(
        dynamic_update=True,
        early_stopping=True,
        seed=1718
    )

    # # Check 1. See if sampler is initialized with correct number of hidden layers
    # assert(sampler.space['n_hidden_layers'] == n_hidden_layers), \
    #     "sampler initialized with incorrect number of hidden layers"

    # # Check 2. See if correct number of hidden unit parameters are defined
    # names      = filter(lambda x: 'n_hidden' in x and 'layers' not in x, sampler.space.keys())
    # cols2check = ['n_hidden%s' % i for i in range(1, n_hidden_layers+1)]
    # assert(set(names) == set(cols2check)), \
    #     "number of hidden unit parameters not defined correctly, check starting space"

    # # Check 3. Make sure sampling changes values within bounds defined by user
    # df_configs = pd.DataFrame([sampler.sample_space() for _ in range(20)])
    # for column in df_configs:
    #     # Remember this hyperparameter should stay fixed
    #     if column == 'n_hidden_layers': 
    #         assert(df_configs[column].unique()[0] == n_hidden_layers), \
    #             "n_hidden_layers should not change during sampling"
    #         continue
        
    #     # Other hyperparameters should change across samples
    #     assert(df_configs[column].unique().shape[0] > 1), \
    #         "sampling generated constant values for %s, check sampling scheme" % column

    # # Check 4. Make sure dynamic updating is working based on made up data
    # df_configs = df_configs.sort_values(by="dropout", ascending=True)
    # df_hof     = df_configs.copy(deep=True).iloc[:5]

    # # Force minimum number of epochs to be greater than 1 for later assertion check
    # df_hof.loc[df_hof['epochs'] == 1, 'epochs'] = 2

    # # Now update space based on 'hof' candidates
    # sampler.update_space(df_hof)
    
    # for column in df_hof:
    #     # These are categorical distributions so skip
    #     if column in ['batch_norm', 'optimizer']: continue
        
    #     actual_min = df_hof[column].min()
    #     actual_max = df_hof[column].max()
    #     hp_bounds  = helper.parse_hyperopt_param(str(sampler.space[column]))[1]

    #     # For necessary hyperparameters, make sure scale is correct
    #     if column in ['learning_rate', 'reg_l1', 'reg_l2']:
    #         actual_min = log(actual_min)
    #         actual_max = log(actual_max)
        
    #     # Since early_stopping is enabled, this parameter should not be dynamically
    #     # updated => the range should still be [1, max_epochs]
    #     if column == 'epochs':
    #         assert(actual_min > hp_bounds[0]), \
    #             "epochs should not be updated with early stopping enabled"
    #         continue

    #     assert(actual_min == hp_bounds[0]), \
    #         "lower bound did not update correctly for %s during space update" % column
        
    #     assert(actual_max == hp_bounds[1]), \
    #         "upper bound did not update correctly for %s during space update" % column
        