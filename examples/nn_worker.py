from keras.layers import Activation, BatchNormalization, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.regularizers import l1, l2

import os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Custom imports
from ptuner import ParallelPipelineTuner, STATUS_FAIL, STATUS_OK
from ptuner.spaces import MLPClassifierSampler, NaiveFeatureSampler, SpaceSampler

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SEED     = 1718


if __name__ == "__main__":

    #############
    # Load data #
    #############

    # Load into memory and split into train/test
    df   = np.loadtxt(os.path.join(DATA_DIR, 'toy_nn.csv'), delimiter=',')
    X, y = df[:, :-1], df[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=.33, random_state=SEED
        )

    # Standardize
    scaler          = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    
    # Define class weights to upweight class 1 vs class 0
    n_class0 = (y_train == 0).sum()
    n_class1 = (y_test == 0).sum()
    class_weight = {
        0 : 1,
        1 : n_class0/n_class1 # every obs in class 1 counts X times that of class 0
    }


    #############################
    # Define objective function #
    #############################

    # Extra params to best updated on different rounds
    def objective(params, current_round):
        """Objective function to be optimized"""

        epochs = [1, 2, 5, 7, 10]

        # Keep imports here to prevent issues with multiprocessing
        # see https://stackoverflow.com/questions/42504669/keras-tensorflow-and-multiprocessing-in-python
        from keras.layers import Activation, BatchNormalization, Dense, Dropout
        from keras.models import Sequential
        from keras.optimizers import Adam, SGD
        from keras.regularizers import l1, l2
        import tensorflow as tf
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

        try:
            # Unpack parameters
            features = params['features']
            hps      = params['nn']

            # Build model
            model = Sequential()
            for i in range(1, hps['n_hidden_layers']+1):
                # Dense layer
                model.add(
                    Dense(units=hps['n_hidden%s' % i], 
                          kernel_regularizer=l2(hps['reg_l2']),
                          activity_regularizer=l1(hps['reg_l1']),
                          input_dim=p if i == 1 else None)
                )   
                # Activation
                model.add(Activation('relu'))

                # Batch norm
                if hps['batch_norm'] == 'yes':
                    model.add(BatchNormalization())

                # Dropout
                model.add(Dropout(hps['dropout']))
            
            # Final layer
            model.add(Dense(units=1, activation='sigmoid'))

            # Get optimizer
            if hps['optimizer'] == 'adam':
                optimizer = Adam(lr=hps['learning_rate'])
            else:
                optimizer = SGD(lr=hps['learning_rate'], nesterov=True)

            # Compile and train
            model.compile(
                loss='binary_crossentropy', 
                optimizer=optimizer,
                metrics=['accuracy']
                )
            model.fit(
                X_train, 
                y_train,
                class_weight=class_weight,
                epochs=epochs[current_round],
                batch_size=hps['batch_size'],
                verbose=0
                )
            
            # Evaluate now
            y_pred = model.predict_proba(X_test)
            try:
                metric = roc_auc_score(y_test, y_pred)
            except:
                metric = 0.0
                
            # Return information, at least containing metric
            return {
                'status'  : STATUS_OK,
                'message' : None,
                'metric'  : metric
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
        role='worker',
        n_jobs=1,
        backend='threading'
        )

    tuner.search(
        objective=objective, 
        max_configs_per_round=[500, 400, 300, 200, 100]
        )