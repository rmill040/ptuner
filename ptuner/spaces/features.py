import logging
import numpy as np 
import pandas as pd
from typing import Any, Dict, List

# Custom imports
from ..base._sampler import BaseSampler

_LOGGER = logging.getLogger(__name__)


class NaiveFeatureSampler(BaseSampler):
    """Feature space sampler using a naive sampling scheme defined as:
        1. Define starting sampling probabilities for each feature
        2. At each round, generate and evaluate X sampled configurations
        3. If dynamic updating is enabled, after best candidates are selected from 
           current round, update each feature's selection probability based on the
           proportion of times the feature was in the best candidates. For example,
           if 20 best configurations are selected and feature X1 appears 15/20 times,
           its selection probability for the next round would be 15/20 = 0.75.
        4. For features below muting threshold, these become muted and the sampler
           will not select them for future configurations
        5. Repeat steps 2-4 for future rounds
    
    Parameters
    ----------
    p : int
        Number of features in original space

    dynamic_update : bool
        Whether to update hyperparameter distributions

    muting_threshold : float
        Threshold for muting features during sampling
    """
    def __init__(
        self, 
        p: int, 
        dynamic_update: bool = True, 
        muting_threshold: float = .30
        ) -> None:

        assert(muting_threshold > 0 and muting_threshold <= 1), \
            "muting threshold (%.2f) should be in range (0, 1]" % muting_threshold
        
        self.p: int                      = p
        self.muting_threshold: np.ndarry = muting_threshold
        self.selection_probs: np.ndarry  = np.array([.5]*self.p)
        self.space: Any                  = self._starting_space()
        super().__init__(dynamic_update=dynamic_update)


    def __str__(self) -> str:
        """Print string representation of class.
        
        Parameters
        ----------
        None

        Returns
        -------
        str
            Name of class
        """
        return "NaiveFeatureSampler"


    def _starting_space(self) -> Any:
        """Create starting space for sampler.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Key/value pairs, key is hyperparameter name and value is whether 
            feature is available to model
        """
        return [True]*self.p


    def sample_space(self) -> Any: 
        """Sample from hyperparameter distributions.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Key/value pairs, key is hyperparameter name and value is whether
            feature is available to model
        """
        selected: np.ndarry  = np.random.binomial(1, self.selection_probs, self.p).astype(bool)
        selected            &= np.array(self.space)
        if not selected.sum(): 
            _LOGGER.warn("no features selected in sampled space, reverting to original space")
            selected = self.space
        return [bool(s) for s in selected]
        

    def update_space(self, data: pd.DataFrame) -> None:
        """Update hyperparameter distributions.

        Parameters
        ----------
        data : pandas DataFrame
            Results of hyperparameter configurations tested

        Returns
        -------
        None
        """
        # Update selection probabilities and find features above muting threshold
        self.selection_probs = np.mean(data.values, axis=0)
        status: np.ndarry    = self.selection_probs > self.muting_threshold

        # Update available features
        updated: np.ndarry = np.array(self.space) & status
        if not updated.sum():
            _LOGGER.warn(
                "no feature scores above muting threshold=%.2f, " % self.muting_threshold + \
                "keeping previous feature set with %d features and ignoring update" % self.space.sum()
                )
            return
        else:
            selected: List[bool] = [bool(u) for u in updated]
            _LOGGER.info(
                "feature sampler updated with %d/%d features available" % \
                    (sum(selected), len(selected))
            )
            self.space = selected
        

# # genetic algorithm for feature selection
# """
#     0. 0/1 bit for feature off/on
#     1. Generate initial population
#     2. Evaluate fitness
#     3. Selection
#         * K-tournament style (default for now)
#     4. Crossover
#         * One point crossover (default for now)
#     5. Mutation
#         * Single bit (default for now)

#     # Add elitism

#     # Add selection rate

#     Misc:
#         - Check for already started run (if query(count) > 0 for example)
# """

# # 'successive half stepping' for hyperparameters
# """
#     0. Initial parameter distributions
#     1. Generate initial population
#     2. 
# """

# class GAFeatureSampler(BaseSampler):
#     """ADD
    
#     Parameters
#     ----------
#     """
#     def __init__(self, p, seed=None):
#         self.p = p
#         super().__init__(seed=seed)


#     def __str__(self):
#         """ADD HERE"""
#         return "GAFeatureSpaceSampler"

    
#     def sample_space(self):
#         pass


#     def update_space(self, data):
#         pass