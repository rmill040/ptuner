from abc import ABC, abstractmethod
from multiprocessing import cpu_count
from typing import Any, List, Optional, Dict

# Package imports
from ..spaces.sampler import SpaceSampler

__all__ = [
    "BasePipelineTuner"
]


class BasePipelineTuner(ABC):
    """Base class for pipeline tuner.
    
    Parameters
    ----------
    lower_is_better : bool
        Whether lower metric indicates better performance

    n_jobs : int
        Number of jobs
    
    backend : str
        Backend for parallel processing using joblib
    
    experiment_name : str
        Name of experiment
    
    save_name : str
        Name of file for saving

    verbose : bool
        Whether to use verbose logging
    """
    @abstractmethod
    def __init__(
        self, 
        lower_is_better: bool, 
        n_jobs: int, 
        backend: str = 'threading',
        experiment_name: str = 'ptuner',
        save_name: Optional[str] = None,
        verbose: bool = True
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
        self.backend: str   = backend
        self.save_name: str = save_name
        self.verbose: bool  = verbose

    
    @abstractmethod
    def _export_all_results(self) -> None:
        """Export all results from search.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass


    @abstractmethod
    def _evaluate_candidate(
        self, 
        objective: Any
        ) -> Any:
        """Evaluate current candidate configuration.

        Parameters
        ----------
        objective : function
            Function to optimize that returns evaluation metric

        Returns
        -------
        dict
            Evaluation results for current candidate configuration
        
        str
            Status of objective function evaluation
        """
        pass


    @abstractmethod
    def _update_space(
        self, 
        sampler: SpaceSampler, 
        hof: int,
        n_round: int
        ) -> SpaceSampler:
        """Updates search space.
        
        Parameters
        ----------    
        sampler : SpaceSampler
            Instantiated SpaceSampler

        hof : int
            Number of top configurations to keep
        
        n_round : int
            Round number for search
        
        Returns
        -------
        sampler : SpaceSampler
            SpaceSampler with updated space distributions
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
        """Run search for best hyperparameters and/or features based on provided 
        objective function.
        
        Parameters
        ----------
        objective : function
            Function to optimize that returns evaluation metric

        sampler : SpaceSampler
            Instantiated SpaceSampler

        max_configs_per_round : list
            List with maximum number of configurations per round

        subsample_factor : int
            Factor used to subsample number of configurations for determining size 
            of hof for current round

        Returns
        -------
        None
        """
        pass