from math import log
import pandas as pd
import pytest
import re

# Package imports
from ptuner.spaces import MLPClassifierSampler, XGBClassifierSampler
from ptuner.utils import helper


@pytest.mark.parametrize("dynamic_update", [True, False])
@pytest.mark.parametrize("early_stopping", [True, False])
@pytest.mark.parametrize("n_hidden_layers", [1, 2])
@pytest.mark.parametrize("max_neurons", [256, 512])
@pytest.mark.parametrize("max_epochs", [10, 25])
def test_mlp_classifier_sampler(
    dynamic_update, 
    early_stopping, 
    n_hidden_layers, 
    max_neurons, 
    max_epochs
    ):
    """Test MLPClassifierSampler.
    """
    # Define sampler
    sampler = MLPClassifierSampler(
        dynamic_update=dynamic_update,
        early_stopping=early_stopping,
        n_hidden_layers=n_hidden_layers,
        max_neurons=max_neurons,
        max_epochs=max_epochs,
        seed=None # No seed
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
    
    # Check 4. Make sure number of hidden units and epochs do not exceed bounds
    # specified by user
    for column in cols2check + ['epochs']:

        max_sampled = max([sampler.sample_space()[column] for _ in range(100)])
        if column == 'epochs':
            assert(max_sampled <= max_epochs), \
                "sampled epoch higher than max epochs, check constraints"
        else:
            assert(max_sampled <= max_neurons), \
                "sampled %s higher than max neurons, check constraints" % column        

    # Check 5. Make sure dynamic updating is working based on made up data
    df_configs = df_configs.sort_values(by="dropout", ascending=True)
    df_hof     = df_configs.copy(deep=True).iloc[:5]

    # Force minimum number of epochs to be greater than 1 for later assertion check
    df_hof.loc[df_hof['epochs'] == 1, 'epochs'] = 2

    # Now update space based on 'hof' candidates
    sampler.update_space(df_hof)
    
    for column in df_hof:
        # These are categorical distributions so skip
        if column in ['batch_norm', 'optimizer', 'n_hidden_layers']: continue
        
        actual_min            = df_hof[column].min()
        actual_max            = df_hof[column].max()
        param_type, hp_bounds = helper.parse_hyperopt_param(str(sampler.space[column]))

        # For necessary hyperparameters, make sure scale is correct
        if param_type == 'loguniform':
            actual_min = log(actual_min)
            actual_max = log(actual_max)
        
        # Handle n_estimators for specific cases
        if column == 'epochs':
            if dynamic_update:

                # With dynamic updates, having early stopping with DISABLE parameter
                # updates
                if early_stopping:
                    assert(actual_min > hp_bounds[0]), \
                    "epochs should not be updated with early stopping enabled, " + \
                    "even with dynamic update enabled"
                    continue  

                # With dynamic updates, not having early stopping will ENABLE parameter
                # updates
                else:
                    assert(actual_min == hp_bounds[0]), \
                        "epochs should be updated with dynamic update and " + \
                        "early stopping disabled"
                    continue
        
        if dynamic_update:
            assert(actual_min == hp_bounds[0]), \
                "lower bound did not update correctly for %s during space update" % column
            
            assert(actual_max == hp_bounds[1]), \
                "upper bound did not update correctly for %s during space update" % column 
        else:
            assert(actual_min >= hp_bounds[0]), \
                "lower bound did not update correctly for %s during space update" % column
            
            assert(actual_max <= hp_bounds[1]), \
                "upper bound did not update correctly for %s during space update" % column


@pytest.mark.parametrize("dynamic_update", [True, False])
@pytest.mark.parametrize("early_stopping", [True, False])
def test_xgb_classifier_sampler(dynamic_update, early_stopping):
    """Tests XGBClassifierSampler.
    """
    # Define sampler
    sampler = XGBClassifierSampler(
        dynamic_update=dynamic_update,
        early_stopping=early_stopping,
        seed=1718
    )

    # Check 1. Make sure sampling changes values within bounds defined by user
    df_configs = pd.DataFrame([sampler.sample_space() for _ in range(20)])
    for column in df_configs:
        # Remember this hyperparameter should stay fixed
        if column == 'random_state': 
            assert(df_configs[column].unique()[0] == 1718), \
                "random_state should not change during sampling"
            continue
        
        # Other hyperparameters should change across samples
        assert(df_configs[column].unique().shape[0] > 1), \
            "sampling generated constant values for %s, check sampling scheme" % column

    # Check 2. Check dynamic updating and early_stopping features if enabled
    df_configs = df_configs.sort_values(by="learning_rate", ascending=True)
    df_hof     = df_configs.copy(deep=True).iloc[:5]

    # Force minimum number of estimators to be greater than 10 for later assertion check
    df_hof.loc[df_hof['n_estimators'] == 10, 'n_estimators'] = 11

    # Now update space based on 'hof' candidates
    sampler.update_space(df_hof)
    
    for column in df_hof:
        # Skip this parameter
        if column == 'random_state': continue

        actual_min            = df_hof[column].min()
        actual_max            = df_hof[column].max()
        param_type, hp_bounds = helper.parse_hyperopt_param(str(sampler.space[column]))

        # For necessary hyperparameters, make sure scale is correct
        if param_type == 'loguniform':
            actual_min = log(actual_min)
            actual_max = log(actual_max)
        
        # Handle n_estimators for specific cases
        if column == 'n_estimators':
            if dynamic_update:

                # With dynamic updates, having early stopping with DISABLE parameter
                # updates
                if early_stopping:
                    assert(actual_min > hp_bounds[0]), \
                    "n_estimators should not be updated with early stopping enabled, " + \
                    "even with dynamic update enabled"
                    continue  

                # With dynamic updates, not having early stopping will ENABLE parameter
                # updates
                else:
                    assert(actual_min == hp_bounds[0]), \
                        "n_estimators should be updated with dynamic update and " + \
                        "early stopping disabled"
                    continue
        
        if dynamic_update:
            assert(actual_min == hp_bounds[0]), \
                "lower bound did not update correctly for %s during space update" % column
            
            assert(actual_max == hp_bounds[1]), \
                "upper bound did not update correctly for %s during space update" % column 
        else:
            assert(actual_min >= hp_bounds[0]), \
                "lower bound did not update correctly for %s during space update" % column
            
            assert(actual_max <= hp_bounds[1]), \
                "upper bound did not update correctly for %s during space update" % column