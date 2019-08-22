import logging
from os.path import join
import sys
from typing import Any

# Custom imports
from ptuner.utils import constants as c
from ptuner.pipeline import (LocalPipelineTuner, ParallelPipelineTuner, 
                             STATUS_FAIL, STATUS_OK)

logging.basicConfig(level=logging.INFO,
                    format=c.FORMAT)

# Create custom logger
logger: Any = logging.getLogger(__name__)

with open(join(c.PACKAGE_ROOT, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()