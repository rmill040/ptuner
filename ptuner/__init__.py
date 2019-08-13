import logging
from os.path import join
import sys

# Custom imports
from ptuner.utils import constants as c
from ptuner.pipeline import LocalPipelineTuner, ParallelPipelineTuner

logging.basicConfig(level=logging.INFO,
                    format=c.FORMAT)

# Create custom logger
logger = logging.getLogger(__name__)

with open(join(c.PACKAGE_ROOT, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()

# Status constants
STATUS_OK   = "OK"
STATUS_FAIL = "FAIL"