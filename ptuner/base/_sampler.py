from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

__all__ = [
    "BaseSampler"
]


class BaseSampler(ABC):
    """Base sampler class.
    
    Parameters
    ----------
    dynamic_update : bool
        Whether to dynamically update search space

    seed : int
        Seed for sampler
    """
    @abstractmethod
    def __init__(
        self, 
        dynamic_update: bool = True, 
        seed: Optional[int] = None
        ) -> None:
        self.space: Any           = self._starting_space()
        self.dynamic_update: bool = dynamic_update
        self.seed: Optional[int]  = seed


    @abstractmethod
    def __str__(self) -> str:
        """Print string representation of class.
        """
        pass


    @abstractmethod
    def _starting_space(self) -> Dict[str, Any]:
        """Create starting space for sampler.
        """
        pass
    

    @abstractmethod
    def sample_space(self) -> Dict[str, Any]:
        """Sample parameters in space.
        """
        pass


    @abstractmethod
    def update_space(self, data: Any) -> None:
        """Update distributions in sample space.
        """
        pass