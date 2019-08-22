from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Dict, Optional

__all__ = [
    "BaseSampler"
]


class BaseSampler(ABC):
    """Base sampler class.
    
    Parameters
    ----------
    dynamic_update : bool
        Whether to update hyperparameter distributions

    seed : int
        Random seed
    """
    @abstractmethod
    def __init__(
        self, 
        dynamic_update: bool = True, 
        seed: Optional[int] = None,
        **kwargs: Dict[Any, Any],
        ) -> None:
        self.space: Any           = self._starting_space()
        self.dynamic_update: bool = dynamic_update
        self.seed: Optional[int]  = seed


    @abstractmethod
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
        pass


    @abstractmethod
    def _starting_space(self) -> Any:
        """Create starting hyperparameter space for sampler.

        Parameters
        ----------
        None

        Returns
        -------
        list - from feature samplers
            List of features to be selected
        dict - from hyperparameter samplers
            Key/value pairs, key is hyperparameter name and value is a sample from
            the hyperparemeter's statistical distribution
        """
        pass
    

    @abstractmethod
    def sample_space(self) -> Any:
        """Sample from hyperparameter distributions.

        Parameters
        ----------
        None

        Returns
        -------
        list - from feature samplers
            List of features to be selected
        dict - from hyperparameter samplers
            Key/value pairs, key is hyperparameter name and value is a sample from
            the hyperparemeter's statistical distribution
        """
        pass


    @abstractmethod
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
        pass