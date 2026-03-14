# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


from ctypes import CDLL, POINTER, RTLD_LOCAL, c_double, c_float, c_int32
from pathlib import Path


# note: size of x should not exceed maximum
def ASINT(x):
    return x.ctypes.data_as(POINTER(c_int32))


def ASDOUBLE(x):
    return x.ctypes.data_as(POINTER(c_double))


def ASFLOAT(x):
    return x.ctypes.data_as(POINTER(c_float))


def register_func(me, dll, name, argtypes=[], restype=None, caller_name=None):
    if caller_name is None:
        caller_name = name
    setattr(me, caller_name, getattr(dll, name))
    func = getattr(me, caller_name)
    func.argtypes = argtypes
    func.restype = restype


_cdll_cache: dict[str, CDLL] = {}


def load_cdll(path):
    """Load a shared library, returning a cached handle if already loaded."""
    key = str(path)
    if key in _cdll_cache:
        return _cdll_cache[key]
    root = Path(__file__).parent.parent.parent
    resolved = root / path
    dll = CDLL(str(resolved), mode=RTLD_LOCAL)
    _cdll_cache[key] = dll
    return dll
