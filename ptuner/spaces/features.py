import numpy as np
import pandas as pd

# Custom imports
from ..base._sampler import BaseSampler


# TODO: ADD FLAG FOR TURNING OFF UPDATES -- ADD DYNAMIC UPDATES
# TODO: ADD LOGGING

class NaiveFeatureSampler(BaseSampler):
    """ADD
    
    Parameters
    ----------
    """
    def __init__(self, p, muting_threshold=.30):
        # TODO: Assertion checking muting_threshold E (0, 1]
        self.p                = p
        self.selection_probs  = np.array([.5]*self.p)
        self.muting_threshold = muting_threshold
        self.space            = self._starting_space()
        super().__init__()


    def __str__(self):
        """ADD HERE"""
        return "NaiveFeatureSampler"


    def _starting_space(self):
        """ADD HERE"""
        return [True]*self.p


    def sample_space(self):
        """ADD HERE"""
        selected  = np.random.binomial(1, self.selection_probs, self.p).astype(bool)
        selected &= np.array(self.space)
        if not selected.sum(): 
            print("[warning] no features selected in sampled space, reverting to original space")
            selected = self.space
        return [bool(s) for s in selected]
        

    def update_space(self, data):
        """ADD HERE"""
        # Update selection probabilities and find features above muting threshold
        self.selection_probs = np.mean(data.values, axis=0)
        status               = self.selection_probs > self.muting_threshold

        # Update available features
        updated = np.array(self.space) & status
        if not updated.sum():
            print(
                "[warning] no feature scores above muting threshold=%.2f," % \
                    self.muting_threshold,
                "keeping previous feature set with %d features and ignoring update" % \
                    self.space.sum()
                )
            return
        else:
            selected = [bool(u) for u in updated]
            print(
                "[info] feature sampler updated with %d/%d features available" % \
                    (sum(selected), len(selected))
            )
            self.space = selected
        

# genetic algorithm for feature selection
"""
    0. 0/1 bit for feature off/on
    1. Generate initial population
    2. Evaluate fitness
    3. Selection
        * K-tournament style (default for now)
    4. Crossover
        * One point crossover (default for now)
    5. Mutation
        * Single bit (default for now)

    # Add elitism

    # Add selection rate

    Misc:
        - Check for already started run (if query(count) > 0 for example)
"""

# 'successive half stepping' for hyperparameters
"""
    0. Initial parameter distributions
    1. Generate initial population
    2. 
"""

class GAFeatureSampler(BaseSampler):
    """ADD
    
    Parameters
    ----------
    """
    def __init__(self, p, seed=None):
        self.p = p
        super().__init__(seed=seed)


    def __str__(self):
        """ADD HERE"""
        return "GAFeatureSpaceSampler"

    
    def sample_space(self):
        pass


    def update_space(self, data):
        pass