from abc import ABC, abstractmethod
from multiprocessing import cpu_count
from typing import Any, List

# Custom imports
from ..spaces import SpaceSampler

__all__ = [
    "BasePipelineTuner"
]


class BasePipelineTuner(ABC):
    """Base class for pipeline tuner.
    
    Parameters
    ----------
    n_jobs : int
        Number of jobs
    
    lower_is_better : bool
        Whether lower metric indicates better performance
    
    experiment_name : str
        Name of experiment
    
    backend : str
        Backend for parallel processing using joblib
    """
    @abstractmethod
    def __init__(
        self, 
        n_jobs: int, 
        lower_is_better: bool, 
        experiment_name: str,
        backend: str = 'threading'
        ) -> None:
        # Calculate number of jobs for parallel processing
        max_cpus: int = cpu_count()
        if n_jobs == 0:
            n_jobs = 1
        elif abs(n_jobs) > max_cpus:
            n_jobs = max_cpus
        else:
            if n_jobs < 0: n_jobs = list(range(1, cpu_count()+1))[n_jobs]
        
        self.n_jobs: int = n_jobs
        
        # Parameters for running experiment
        self.lower_is_better: bool = lower_is_better
        self.experiment_name: str  = experiment_name
        if backend not in ['loky', 'threading', 'multiprocessing']:
            raise ValueError(
                "backend (%s) not a valid argument, " % backend, 
                "use loky, threading, or multiprocessing"
            )
        self.backend: str = backend

    
    @abstractmethod
    def _export_all_results(self) -> None:
        """Export all results from search.
        """
        pass


    @abstractmethod
    def _evaluate_candidate(self, objective: Any) -> Any:
        """Evaluate current candidate configuration.
        """
        pass


    @abstractmethod
    def _update_space(
        self, 
        sampler: SpaceSampler, 
        hof: int,
        n_round: int
        ) -> SpaceSampler:
        """Update sampler space.
        """
        pass


    @abstractmethod
    def search( 
        self, 
        objective: Any, 
        sampler: SpaceSampler, 
        max_configs_per_round: List[int], 
        subsample_factor: int
        ) -> None: 
        """Begin dynamic random search.
        """
        pass