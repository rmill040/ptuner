from joblib import delayed, Parallel
import json
import logging
from multiprocessing import cpu_count, Manager
import numpy as np
import pandas as pd
import time
from tqdm import trange
from typing import Any, cast, Dict, List, Optional, Tuple

# Custom imports
from .base._pipeline import BasePipelineTuner
from .db import init_collection, is_running, MongoError, MongoWorker
from .spaces import SpaceSampler 
from .utils.helper import countdown, get_hostname

_LOGGER = logging.getLogger(__name__)


class LocalPipelineTuner(BasePipelineTuner):
    """Simple local pipeline tuner for both hyperparameter tuning and feature selection.
    
    Parameters
    ----------
    lower_is_better : bool
        Whether lower metrics indicate better performance
    
    experiment_name : str
        Name of experiment that will be the collections name in MongoDB
    
    n_jobs : int
        Number of parallel workers
    """
    def __init__(
        self, 
        lower_is_better: bool,
        experiment_name: str = 'ptuner', 
        n_jobs: int = -1,
        backend: str = 'threading',
        save_name: Optional[str] = None
        ) -> None:

        super().__init__(                       
            lower_is_better=lower_is_better, 
            experiment_name=experiment_name,
            n_jobs=n_jobs,
            backend=backend
            )

        self.history: List[Any]           = []
        self.best_results: Dict[str, Any] = Manager().dict()
        self.best_results['metric']       = \
            np.inf if self.lower_is_better else -np.inf
        self.best_results['params']       = None
        
        # Name of results file
        self.save_name = save_name if save_name else experiment_name + '.csv'


    def _export_all_results(self) -> None:
        """Export all search results to .csv file.
        """
        try:
            pd.concat([pd.DataFrame(h) for h in self.history], axis=0)\
                .sort_values('metric', ascending=self.lower_is_better)\
                .to_csv(self.save_name, index=False)
            _LOGGER.info("exported all results to disk at %s" % self.save_name)
        except Exception as e:
            _LOGGER.error("error exporting all results because %s" % e)


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
        # Create dataframe for current round and select top performing config
        df_all = pd.concat([pd.DataFrame(h) for h in self.history], axis=0)
        df_all = df_all[df_all['round'] == n_round]\
                    .sort_values(by='metric', ascending=self.lower_is_better)[:hof]

        # Update feature space
        if sampler.feature_sampler:
            df_features = pd.DataFrame.from_records(
                df_all['params'].apply(lambda x: x[sampler.feature_sampler.name]).values.tolist()
            )
            sampler.feature_sampler.sampler.update_space(df_features)

        # Update hyperparameters space
        if sampler.hyperparameter_samplers:
            for s in sampler.hyperparameter_samplers:
                df_hyperparameters = pd.DataFrame.from_records(
                    df_all['params'].apply(lambda x: x[s.name]).values.tolist()
                    )
                s.sampler.update_space(df_hyperparameters)
                
        return sampler

    
    def _evaluate_candidate( #type: ignore
        self, 
        objective: Any, 
        candidate: Dict[str, Any], 
        i: int, 
        n_candidates: int
        ) -> Dict[str, Any]:
        """Evaluates candidate configuration using objective function.
        
        Parameters
        ----------
        objective : function
            Function to optimize that returns evaluation metric

        candidate : dict
            Contains parameters for the objective function and bookkeeping for
            current round and configuration number

        i : int
            Number of candidate in current round

        n_candidates : int
            Total number of candidates to evaluate in current round
        
        Returns
        -------
        results : dict
            Evaluation results for current candidate configuration
        """
        if i % self.n_jobs == 0:
            _LOGGER.info("evaluating candidates, %d remaining" % (n_candidates-i))

        results: Any = objective(
            params=candidate['params'], 
            current_round=candidate['round']-1
            )

        # Check if failure occurred during objective func evaluation
        if 'FAIL' in results['status'].upper() or 'OK' not in results['status'].upper():
            msg: str = "running candidate failed"
            if 'message' in results.keys():
                if results['message']: msg += ' because %s' % results['message']
            _LOGGER.warn(msg)
            return {
                'status' : results['status'],
                'metric' : np.inf if self.lower_is_better else -np.inf,
                'params' : candidate['params'],
                'round'  : candidate['round'],
                'config' : candidate['config']
            }
        
        # Find best metric so far and compare results to see if current result is better
        if self.lower_is_better:
            if results['metric'] < self.best_results['metric']:
                _LOGGER.info("new best metric %.4f" % results['metric'])
                self.best_results['metric'] = results['metric']
                self.best_results['params'] = candidate['params']        
        else:
            if results['metric'] > self.best_results['metric']:
                _LOGGER.info("new best metric %.4f" % results['metric'])
                self.best_results['metric'] = results['metric']
                self.best_results['params'] = candidate['params']        
 
        return {
            'status' : results['status'],
            'metric' : results['metric'],
            'params' : candidate['params'],
            'round'  : candidate['round'],
            'config' : candidate['config']
            }


    def search(
        self, 
        objective: Any, 
        sampler: SpaceSampler, 
        max_configs_per_round: List[int],
        subsample_factor: int = 5
        ) -> None:
        """Run search for best hyperparameters and/or features based on provided 
        objective function.
        
        Parameters
        ----------
        objective : function
            Function to optimize that returns evaluation metric

        Returns
        -------
        None
        """
        _LOGGER.info("beginning search with %d jobs using %s backend" % \
            (self.n_jobs, self.backend)
            )
        start: float = time.time()

        # Begin search
        n_rounds: int = len(max_configs_per_round)
        for n, n_configs in enumerate(max_configs_per_round):
            
            # Grab hof candidates for this round
            hof: int = int(np.ceil(n_configs/subsample_factor))
            
            # Zero indexed so handle this now
            n += 1
            _LOGGER.info("beginning round %d/%d" % (n, n_rounds))

            # Sample space
            candidates: List[Any] = []
            with trange(n_configs) as t:
                for i in t:
                    t.set_description('generating candidate %s/%s' % (i+1, n_configs))
                    candidates.append({
                        'params' : sampler.sample_space(),
                        'round'  : n,
                        'config' : i+1
                    })
            n_candidates: int = len(candidates)
            _LOGGER.info("generated %d candidates for evaluation" % len(candidates))

            # Begin evaluating candidates in parallel
            if n_candidates:
                _LOGGER.info("evaluating candidates, %d remaining" % n_candidates)
                output = Parallel(n_jobs=self.n_jobs, verbose=False, backend=self.backend)\
                    (delayed(self._evaluate_candidate)(objective, candidate, i+1, n_candidates)\
                        for i, candidate in enumerate(candidates)
                        )
                self.history.append(output)

            # Update search space now
            if n < n_rounds:
                _LOGGER.info("updating search space")
                sampler = self._update_space(sampler, hof, n_round=n)

        # Export all results
        self._export_all_results()

        total_time: float = (time.time() - start) / 60.
        _LOGGER.info("finished search in %.2f minutes" % total_time)


class ParallelPipelineTuner(BasePipelineTuner):
    """Simple distributed pipeline tuner for both hyperparameter tuning and 
    feature selection using MongoDB as a backend.
    
    Parameters
    ----------
    db_host : str
        IP address of MongoDB instance
    
    db_port : int
        Port for MongoDB instance

    lower_is_better : bool
        Whether lower metrics indicate better performance
    
    experiment_name : str
        Name of experiment that will be the collections name in MongoDB
    
    role : str
        Master or worker
    
    n_jobs : int
        Number of parallel workers
    """
    def __init__(
        self, 
        db_host: str, 
        db_port: int, 
        lower_is_better: bool,
        experiment_name: str ='ptuner', 
        role: str ='master',
        n_jobs: int = -1,
        backend: str = 'threading'
        ) -> None:

        super().__init__(                       
            lower_is_better=lower_is_better, 
            experiment_name=experiment_name,
            n_jobs=n_jobs,
            backend=backend
            )

        # Dictionary to hold best results after searching complete
        self.best_results: Dict[str, Any] = {}

        # Location of MongoDB instance
        self.db_host: str = db_host
        self.db_port: int = int(db_port)

        # Set MongoDB connection params
        self._mongo_params: Dict[str, Any] = {
            'host' : self.db_host,
            'port' : self.db_port
        }
        
        # Parameters for running experiment
        self.role: str            = role
        self.candidates_name: str = self.experiment_name + '_' + 'candidates'

        # Get computer hostname
        self.computer_name: str = get_hostname()
        _LOGGER.info("starting %s on host %s" % (role, self.computer_name))

        # Check if MongoDB is running on host and port
        status, info = is_running(**self._mongo_params)
        msg: str     = ''
        if not status:
            msg = "unable to connect to MongoDB on {host}:{port} because {reason}"\
                    .format(**self._mongo_params, reason=info)
            _LOGGER.error(msg)
            raise MongoError(msg)
        _LOGGER.info(info)
        
        overwrite = True if self.role.lower().strip() == 'master' else False
        
        # Initialize MongoDB DB and collection for holding results
        status, info = init_collection(
            collection=self.experiment_name,
            overwrite=overwrite,
            computer_name=self.computer_name,
            **self._mongo_params
            )
        if not status:
            msg = "unable to initialize MongoDB collection {col} because {reason}"\
                    .format(col=experiment_name, reason=info)
            _LOGGER.error(msg)
            raise MongoError(msg)
        _LOGGER.info(info)

        # Initialize MongoDB DB and collection for holding results
        status, info = init_collection( #type: ignore
            **self._mongo_params,
            collection=self.candidates_name,
            overwrite=overwrite,
            computer_name=self.computer_name
            )
        if not status:
            msg = "unable to initialize MongoDB collection {col} because {reason}"\
                    .format(col=experiment_name, reason=info)
            _LOGGER.error(msg)
            raise MongoError(msg)
        _LOGGER.info(info)


    def _export_all_results(self) -> None:
        """Export all search results to .csv file.        
        """
        with MongoWorker(collection=self.experiment_name, **self._mongo_params) as db: 
            pd.DataFrame.from_records(db.find())\
                .drop('_id', axis=1)\
                .to_csv(self.experiment_name + '.csv', index=False)
        _LOGGER.info("results written to %s.csv" % self.experiment_name)
    

    def _select_best(self) -> Tuple[Any, Any]:
        """Selects top metric and configuration so far in the search.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        metric : float
            Metric calculated in objective function
        
        params : dict
            Parameters specified in sampler
        """
        with MongoWorker(collection=self.experiment_name, **self._mongo_params) as db: 
            query: Any = db.find_one(sort=[("metric", 1 if self.lower_is_better else -1)])
            if 'metric' not in query.keys():
                return None, None
            else:
                return query['metric'], query['params']


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
            Number of top candidates to keep during update
        
        n_round : int
            Round for current results
        
        Returns
        -------
        sampler : SpaceSampler
            Updated SpaceSampler
        """
        with MongoWorker(collection=self.experiment_name, **self._mongo_params) as db: 
            query: List[Any] = list(db.aggregate([
                {"$sort": {"metric": -1}},
                {"$limit": hof},
                {"$match": {"round": {"$eq": n_round}}}
            ]))
        
        # Create dataframe
        df_all: pd.DataFrame = pd.DataFrame(query)

        # Update feature space
        if sampler.feature_sampler:
            df_features: pd.DataFrame = pd.DataFrame.from_records(
                df_all['params'].apply(lambda x: x[sampler.feature_sampler.name]).values.tolist()
            )
            sampler.feature_sampler.sampler.update_space(df_features)

        # Update hyperparameters space
        if sampler.hyperparameter_samplers:
            for s in sampler.hyperparameter_samplers:
                df_hyperparameters: pd.DataFrame = pd.DataFrame.from_records(
                    df_all['params'].apply(lambda x: x[s.name]).values.tolist()
                    )
                s.sampler.update_space(df_hyperparameters)
                
        return sampler


    def _current_round(self) -> int:
        """Query MongoDB to get current round of evaluation.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        round : int
            Current round for evaluation
        """
        with MongoWorker(collection=self.experiment_name, **self._mongo_params) as db: 
            try:
                return cast(int, db.find_one(sort=[("round", -1)])["round"])
            except:
                return -1


    def _n_candidates_remaining(self) -> int:
        """Query MongoDB to find number of candidates remaining for evaluation.
        
        Parameters
        ----------
        None

        Returns
        -------
        n_candidates : int
            Number of candidate configurations remaining in current round for 
            evaluation
        """
        with MongoWorker(collection=self.candidates_name, **self._mongo_params) as db: 
            n_count: int = db.count()
            if n_count == 1:
                if 'message' in db.find_one().keys(): n_count -= 1
            return n_count
    

    def _evaluate_candidate(self, objective: Any) -> str:
        """Evaluates candidate configuration using objective function.
        
        Parameters
        ----------
        objective : function
            Function to optimize that returns evaluation metric

        Returns
        -------
        status : str
            Status of objective function evaluation
        """
        with MongoWorker(collection=None, **self._mongo_params) as db: 
            # Select candidate
            db_candidates: Any = db[self.candidates_name]
            query: Any         = db_candidates.find_and_modify(
                query={"round": {"$exists": True}}, remove=True
                )

            # Query could be empty
            if not query: 
                _LOGGER.warn("query is empty, no configuration available to test")
                return "EMPTY"

            # Evaluate candidate
            db_experiment: Any      = db[self.experiment_name]
            results: Dict[str, Any] = objective(
                params=query['params'], current_round=query['round']-1 # zero-indexed
                )

            # Check if failure occurred during objective func evaluation
            if 'FAIL' in results['status'].upper() or 'OK' not in results['status'].upper():
                msg: str = "running candidate failed"
                if 'message' in results.keys():
                    if results['message']: msg += ' because %s' % results['message']
                _LOGGER.warn(msg)
                return "FAIL"
            
            # Find best metric so far and compare results to see if current result is better
            best_metric, best_params = self._select_best()  
            if best_metric:
                if self.lower_is_better:
                    if results['metric'] < best_metric:
                        _LOGGER.info("new best metric %.4f" % results['metric'])
                        _LOGGER.info("new best params\n%s" % best_params)
                else:
                    if results['metric'] > best_metric:
                        _LOGGER.info("new best metric %.4f" % results['metric'])
                        _LOGGER.info("new best params\n%s" % best_params)
            else:
                # Nothing has been written to db, so just print first result
                _LOGGER.info("new best metric %.4f" % results['metric'])

            db_experiment.insert_one({
                'status' : results['status'],
                'metric' : results['metric'],
                'params' : query['params'],
                'round'  : query['round'],
                'config' : query['config']
            })
            return "OK"


    def _master_search(
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
        # Begin search
        n_rounds: int = len(max_configs_per_round)
        for n, n_configs in enumerate(max_configs_per_round):
            
            # Grab hof candidates for this round
            hof: int = int(np.ceil(n_configs/subsample_factor))
            
            # Zero indexed so handle this now
            n += 1
            _LOGGER.info("beginning round %d/%d" % (n, n_rounds))

            # Drop initialization messages in collections, we are not able to drop
            # the record before other records are written as it will delete the 
            # entire collection so we wait until round 2 to drop anything
            if n == 2:
                with MongoWorker(collection=self.candidates_name, **self._mongo_params) as db: 
                    db.find_and_modify(
                        query={"message" : {"$exists": True}}, remove=True
                        )
                with MongoWorker(collection=self.experiment_name, **self._mongo_params) as db: 
                    db.find_and_modify(
                        query={"message" : {"$exists": True}}, remove=True
                        )

            # Sample space and push to MongoDB
            candidates: List[Any] = []
            with trange(n_configs) as t:
                for i in t:
                    t.set_description('generating candidate %s/%s' % (i+1, n_configs))
                    candidates.append({
                        'params' : sampler.sample_space(),
                        'round'  : n,
                        'config' : i+1
                    })
            with MongoWorker(collection=self.candidates_name, **self._mongo_params) as db: 
                db.insert_many(candidates)
                _LOGGER.info("generated %d candidates for evaluation" % len(candidates))

            # Begin evaluating candidates in parallel
            n_candidates: int = self._n_candidates_remaining()
            if n_candidates:
                while n_candidates:
                    _LOGGER.info("evaluating candidates, %d remaining" % n_candidates)
                    output: List[str] = Parallel(n_jobs=self.n_jobs, verbose=False, backend=self.backend)\
                        (delayed(self._evaluate_candidate)(objective)\
                            for _ in range(self.n_jobs) 
                            )
                    if 'EMPTY' in output: 
                        _LOGGER.info("no candidates remaining to evaluate on master")
                        break
                    n_candidates = self._n_candidates_remaining()
            
            # Update search space now
            if n < n_rounds:
                _LOGGER.info("updating search space")
                sampler = self._update_space(sampler, hof, n_round=n)


    def _worker_search(
        self, 
        objective: Any, 
        n_rounds: int, 
        max_attempts: int = 60, 
        backoff_period: int = 10
        ) -> None:
        """Run search for best hyperparameters and/or features based on provided 
        objective function as a worker role.
        
        Parameters
        ----------
        objective : function
            Function to optimize that returns evaluation metric

        n_rounds : int
            Total number of rounds for evaluation during search
        
        max_attempts : int
            Max attempts for worker to try and poll for candidates to evaluate during
            parallel search
        
        backoff_period : int
            Time in seconds for worker to wait before polling for candidates to 
            evaluate during parallel search

        Returns
        -------
        None
        """
        n_candidates: int  = self._n_candidates_remaining()
        current_round: int = self._current_round()
        if current_round == -1:
            _LOGGER.error("no master instance running, quitting program")
            return
        if not n_candidates and current_round == n_rounds:
            _LOGGER.info("no available candidates to evaluate in final round")
            return
        
        finished: bool = False
        while not finished:
            if n_candidates:
                while n_candidates:
                    _LOGGER.info("evaluating candidates, %d remaining" % n_candidates)
                    output: List[Any] = Parallel(n_jobs=self.n_jobs, verbose=False, backend=self.backend)\
                        (delayed(self._evaluate_candidate)(objective)\
                            for _ in range(self.n_jobs) 
                            )
                    if 'EMPTY' in output: 
                        _LOGGER.info("no candidates remaining to evaluate on worker")
                        break
                    n_candidates = self._n_candidates_remaining()
            else:
                current_round = self._current_round()

                # Keep checking until more jobs become available
                if current_round < n_rounds:
                    _LOGGER.info(
                        "no candidates found in round %d, waiting for next round" \
                            % current_round
                    )
                    attempts: int = 1
                    while not n_candidates:
                        _LOGGER.info("waiting period %d to check for new candidates" % attempts)
                        countdown(message="checking for new candidates in", t=backoff_period)
                        n_candidates = self._n_candidates_remaining()
                        
                        attempts += 1
                        if attempts > max_attempts:
                            msg: str = "max attempts %d exceed, check " % max_attempts
                            msg     += "to see if master failed"
                            _LOGGER.warning(msg)
                            finished = True
                            break
                
                # Search finished
                else:
                    finished = True
                    break


    def search( #type: ignore
        self, 
        objective: Any, 
        max_configs_per_round: List[int], 
        sampler: Any, 
        subsample_factor: int = 5,
        max_attempts: int = 60,
        backoff_period: int = 10
        ) -> None:
        """Run search for best hyperparameters and/or features based on provided 
        objective function.
        
        Parameters
        ----------
        objective : function
            Function to optimize that returns evaluation metric

        max_configs_per_round : list
            List with maximum number of configurations per round
        
        sampler : SpaceSampler
            Instantiated SpaceSampler

        subsample_factor : int
            Factor used to subsample number of configurations for determining size 
            of hof for current round

        max_attempts : int
            Max attempts for worker to try and poll for candidates to evaluate during
            parallel search
        
        backoff_period : int
            Time in seconds for worker to wait before polling for candidates to 
            evaluate during parallel search

        Returns
        -------
        None
        """
        _LOGGER.info("beginning search as %s role with %d jobs using %s backend" % \
            (self.role, self.n_jobs, self.backend)
            )

        start: float = time.time()

        if self.role == 'master':
            self._master_search(
                objective=objective, 
                sampler=sampler, 
                max_configs_per_round=max_configs_per_round,
                subsample_factor=subsample_factor
                )
        else:
            self._worker_search(
                objective=objective, 
                n_rounds=len(max_configs_per_round),
                max_attempts=60,
                backoff_period=backoff_period
                )
        
        self._export_all_results()

        # Query for best results
        self.best_results['metric'], self.best_results['params'] = self._select_best()

        total_time: float = (time.time() - start) / 60.
        _LOGGER.info("finished search in %.2f minutes" % total_time)