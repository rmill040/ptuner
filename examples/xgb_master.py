import numpy as np
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Custom imports
from ptuner import ParallelPipelineTuner, STATUS_FAIL, STATUS_OK
from ptuner.spaces import NaiveFeatureSampler, SpaceSampler, XGBClassifierSampler

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SEED     = 1718

def make_data(N, p):
    """Generate toy data."""
    X = np.random.normal(0, 1, (N, p))
    y = 2 + .59*X[:, 0] + .39*X[:, 1] + .39*X[:, 2]
    y = 1 / (1 + np.exp(-y))
    y = np.random.binomial(1, y, N)
    np.savetxt(os.path.join(DATA_DIR, 'toy_xgb.csv'), np.column_stack([X, y]), delimiter=',')
    

if __name__ == "__main__":

    np.random.seed(SEED)

    #############
    # Make data #
    #############

    N, p = 500, 100
    make_data(N, p)

    # Load into memory and split into train/test
    df   = np.loadtxt(os.path.join(DATA_DIR, 'toy_xgb.csv'), delimiter=',')
    X, y = df[:, :-1], df[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=.33, random_state=SEED
        )


    ##################
    # Define sampler #
    ##################
    
    sampler = SpaceSampler()
    sampler.add_feature_sampler(
        name='features',
        sampler=NaiveFeatureSampler(p=p)
    )
    sampler.add_hyperparameter_sampler(
        name='xgboost',
        sampler=XGBClassifierSampler(early_stopping=True)
    )


    #############################
    # Define objective function #
    #############################


    def objective(params, current_round):
        """Objective function to be optimized"""

        # Extra params to best updated on different rounds
        fit_params = {
            'eval_set'              : None,
            'eval_metric'           : 'auc',
            'early_stopping_rounds' : None,
            'verbose'               : False
            }

        early_stopping_rounds = [10, 50, 100, 250, 500]

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
            
            # Return information, at least containing metric
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

    ################
    # Define tuner #
    ################

    tuner = ParallelPipelineTuner(
        db_host='localhost', 
        db_port=27017, 
        lower_is_better=False,
        experiment_name='ptuner', 
        role='master',
        n_jobs=-1,
        backend='threading'
        )

    tuner.search(
        objective=objective, 
        sampler=sampler, 
        max_configs_per_round=[500, 400, 300, 200, 100]
        )