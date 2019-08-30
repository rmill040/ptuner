import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Package imports
from ptuner import LocalPipelineTuner, ParallelPipelineTuner, STATUS_FAIL, STATUS_OK
from ptuner.spaces import NaiveFeatureSampler, SpaceSampler, XGBClassifierSampler

# Constants
N               = 250
p               = 20
FEATURE_SAMPLER = 'features'
XGB_SAMPLER     = 'xgboost'

# Define sampler
sampler = SpaceSampler()
sampler.add_feature_sampler(
    name=FEATURE_SAMPLER,
    sampler=NaiveFeatureSampler(p=p)
)
sampler.add_hyperparameter_sampler(
    name=XGB_SAMPLER,
    sampler=XGBClassifierSampler(early_stopping=True)
)

@pytest.fixture
def clf_binary():
    """Generate toy data for binary classification.
    """
    X, y = make_classification(
        n_samples=N, 
        n_features=p, 
        n_informative=10,
        n_redundant=10,
        n_classes=2,
        class_sep=1.0,
        random_state=1718
        )
    
    # Train/test split
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, stratify=y, test_size=.33, random_state=1718)
    return X_train, X_test, y_train, y_test


def test_local_pipeline_tuner(clf_binary):
    """Run end-to-end test for LocalPipelineTuner.
    """
    def objective(params, current_round):
        """Objective function to be optimized"""

        # Extra params to best updated on different rounds
        fit_params = {
            'eval_set'              : None,
            'eval_metric'           : 'auc',
            'early_stopping_rounds' : None,
            'verbose'               : False
            }

        early_stopping_rounds = [10, 50, 100]

        try:
            # Unpack parameters
            features = params['features']
            hps      = params['xgboost']

            # Update fit params
            fit_params['eval_set']              = [(X_test[:, features], y_test)]
            fit_params['early_stopping_rounds'] = early_stopping_rounds[current_round]

            # Train/evaluate model
            clf = XGBClassifier(**hps).fit(
                X_train[:, features], 
                y_train, 
                **fit_params
                )

            # Update n_estimators because of early stopping
            params['xgboost']['n_estimators'] = clf.best_iteration+1
            
            # Return information
            return {
                'status'  : STATUS_OK,
                'message' : None,
                'metric'  : clf.best_score
            }
        except Exception as e:
            return {
                'status'  : STATUS_FAIL,
                'message' : e,
                'metric'  : 0.0
            }

    X_train, X_test, y_train, y_test = clf_binary

    tuner = LocalPipelineTuner(
        lower_is_better=False,
        n_jobs=-1,
        backend='threading',
        experiment_name='local_test',
        save_name=None, # Turn off saving
        verbose=False
    )
    tuner.search(
        objective=objective,
        sampler=sampler,
        max_configs_per_round=[75, 50, 25]
    )

    # Check 1. Make sure names of feature sampler and xgboost sampler
    # are available by name in search results
    param_keys = list(tuner.best_results['params'].keys())
    assert(set(param_keys) == set([FEATURE_SAMPLER, XGB_SAMPLER])), \
        "error selecting samplers by their user-defined names"

    # Check 2. Make sure AUC score is not complete garbage in a simple binary
    # classification data set
    assert(tuner.best_results['metric'] > .90), \
        "AUC metric is too low, search algorithm may not be working correctly"

    # Check 3. Make sure all features are not selected
    assert(sum(tuner.best_results['params']['features']) < p), \
        "too many features selected, feature selection algorithm may not be " + \
        "working correctly"


def test_parallel_pipeline_tuner(clf_binary):
    """Run end-to-end test for ParallelPipelineTuner.
    """
    # TODO: Add test
    pass