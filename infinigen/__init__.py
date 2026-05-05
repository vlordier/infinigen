# Copyright (C) 2023, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

import logging
import warnings
from pathlib import Path

__version__ = "1.19.1"

__all__ = ["__version__", "repo_root"]

# Silence landlab's use of the deprecated Dataset.dims API (removed in future xarray versions)
warnings.filterwarnings(
    "ignore",
    message=r".*Dataset\.dims.*",
    category=FutureWarning,
)


def repo_root():
    return Path(__file__).parent.parent
