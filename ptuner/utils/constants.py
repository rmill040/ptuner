import pathlib
from typing import Any

# Custom imports
import ptuner

DB_NAME: str = "db_foo"

# Format for logger
FORMAT: str = '[%(asctime)s] %(levelname)s - %(message)s'

# Package root
PACKAGE_ROOT: Any = pathlib.Path(ptuner.__file__).resolve().parent