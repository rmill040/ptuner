from collections import namedtuple
import numpy as np
from typing import Any, Dict, List, Optional

__all__ = [
    "SpaceSampler"
]

# TODO: ADD LOGGER HERE

NAMED_SAMPLER: namedtuple = namedtuple("Sampler", "name sampler")

class SpaceSampler:
    """ADD
    
    Parameters
    ----------
    """
    def __init__(self) -> None:
        self.feature_sampler: Optional[Any]     = None
        self.hyperparameter_samplers: List[Any] = []
        self._initialized: bool                 = False


    def add_feature_sampler(self, sampler: Any, name: str) -> None:
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        self.feature_sampler = NAMED_SAMPLER(name=name, sampler=sampler)
        self._initialized    = True
    

    def add_hyperparameter_sampler(self, sampler: Any, name: str) -> None:
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        self.hyperparameter_samplers.append(
            NAMED_SAMPLER(name=name, sampler=sampler)
        )
        self._initialized = True


    def sample_space(self) -> Dict[str, Any]:
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        assert(self._initialized), "[error] no samplers detected"

        # Sampler spaces
        params: Dict[str, Any] = {}
        if self.feature_sampler:
            params[self.feature_sampler.name] = \
                self.feature_sampler.sampler.sample_space()
        if self.hyperparameter_samplers:
            for s in self.hyperparameter_samplers:
                params[s.name] = s.sampler.sample_space()
        return params


    def update_space(
        self, 
        data_features:Optional[Any], 
        data_hyperparameters:Optional[Any]
        ) -> None:
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        assert(self._initialized), "[error] no samplers detected"

        # Update spaces
        if self.feature_sampler:
            self.feature_sampler.sampler.update_space(data_features)
        if self.hyperparameter_samplers:
            for s in self.hyperparameter_samplers:
                s.sampler.update_space(data_hyperparameters)