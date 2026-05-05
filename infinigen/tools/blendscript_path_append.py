# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import os
import sys
import warnings

# Silence landlab's deprecated Dataset.dims API (removed in future xarray versions)
warnings.filterwarnings(
    "ignore",
    message=r".*Dataset\.dims.*",
    category=FutureWarning,
)

pwd = os.getcwd()
sys.path.append(pwd)
