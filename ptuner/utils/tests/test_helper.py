from hyperopt import hp
from math import log
import pytest

# Package imports
from ptuner.utils.helper import parse_hyperopt_param

@pytest.mark.parametrize("parameter, expected", [
    (hp.loguniform('test', log(1e-4), log(1)), ('loguniform', [log(1e-4), log(1)])),
    (hp.quniform('test', 1, 100, 1), ('quniform', [1, 100])),
    (hp.uniform('test', 0, 1), ('uniform', [0, 1]))
])
def test_parse_hyperopt_param(parameter, expected):
    """Test valid input to parse_hyperopt_param.
    """
    string                = str(parameter)
    param_type, hp_bounds = parse_hyperopt_param(string)
    assert(param_type == expected[0] and hp_bounds == expected[1]), \
        "error parsing hyperopt parameter"

def test_parse_hyperopt_param_error():
    """Test invalid input to parse_hyperopt_param.
    """
    # This will raise an error since pchoice is a categorical distribution
    parameter = hp.pchoice('test', [(0.5, 'no'), (0.5, 'yes')])
    with pytest.raises(ValueError):
        parse_hyperopt_param(str(parameter))
