from collections import namedtuple
import numpy as np
import re
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from typing import Any, Dict, List, Optional


__all__ = [
    "SpaceSampler"
]

# 1. Gather initial samples
# 2. For round > 1, fit GPR and select optimal points maximizing acquisition function
# 3. Consider adjusting exploration vs exploitation parameter as search continues



# class GPCandidateSelector:
#     """Gaussian process candidate selector -- ADD MORE LATER.

#     Parameters
#     ----------
#     """
#     def __init__(self, kernel=None):
#         pass

#         # if kernel is None:
#         #     kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
#         # self.gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise**2)


#     def _parse_hyperopt_param(self, string: str) -> Any:
#         """ADD
        
#         Parameters
#         ----------
        
#         Returns
#         -------
#         """
#         param_type: str = re.findall('categorical|quniform|uniform|loguniform', string)[0]
#         if param_type == 'categorical':
#             raise ValueError("categorical bounds not supported")
#         else:
#             return param_type, list(map(float, re.findall("[Literal{}](-*\d+\.*\d*)", string)[:2]))


#     def select_best_points(self, sampler: SpaceSampler, n: int):
#         """ADD
        
#         Parameters
#         ----------
        
#         Returns
#         -------
#         """
#         pass
        
    

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
        return self
    

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
        return self


    def sample_space(self) -> Dict[str, Any]:
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        assert(self._initialized), "no samplers detected"

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
        data_features: Optional[Any], 
        data_hyperparameters: Optional[Any]
        ) -> None:
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        assert(self._initialized), "no samplers detected"

        # Update spaces
        if self.feature_sampler:
            self.feature_sampler.sampler.update_space(data_features)
        if self.hyperparameter_samplers:
            for s in self.hyperparameter_samplers:
                s.sampler.update_space(data_hyperparameters)


# if __name__ == "__main__":
#     from hyperopt import hp
#     from math import log
#     selector = GPCandidateSelector()
#     for param in [
#         hp.uniform('xgb_cbl', 0.20, 1.0),
#         hp.loguniform('xgb_ra', log(1e-10), log(1e-1)),
#         hp.quniform('xgb_mds', 0, 3, 1)
#         ]:
#         print(selector._parse_hyperopt_param(param.__str__()))