import numpy as np    
import scipy

class np_backend:
    """A shim that allows a backend to be swapped at runtime."""
    def __init__(self, src):
        self._srcmodule = src

    def __getattr__(self, key):
        if key == '_srcmodule':
            return self._srcmodule

        return getattr(self._srcmodule, key)
    
class scipy_backend:
    """A shim that allows a backend to be swapped at runtime."""
    def __init__(self, src):
        self._srcmodule = src

    def __getattr__(self, key):
        if key == '_srcmodule':
            return self._srcmodule

        return getattr(self._srcmodule, key)
    
try:
    import cupy as cp
    import cupyx.scipy
    cp.cuda.Device(0).compute_capability
    cupy_avail = True
except ImportError:
    cupy_avail = False
    
xp = np_backend(cp) if cupy_avail else np_backend(np)
_scipy = scipy_backend(cupyx.scipy) if cupy_avail else scipy_backend(scipy)

def update_np(module):
    xp._srcmodule = module
    
def update_scipy(module):
    _scipy._srcmodule = module
    
def ensure_np_array(arr):
    if cp and isinstance(arr, cp.ndarray):
        return arr.get()
    else:
        return arr

def pad_or_crop( arr_in, npix ):
    n_arr_in = arr_in.shape[0]
    if n_arr_in == npix:
        return arr_in
    elif npix < n_arr_in:
        x1 = n_arr_in // 2 - npix // 2
        x2 = x1 + npix
        arr_out = arr_in[x1:x2,x1:x2].copy()
    else:
        arr_out = xp.zeros((npix,npix), dtype=arr_in.dtype)
        x1 = npix // 2 - n_arr_in // 2
        x2 = x1 + n_arr_in
        arr_out[x1:x2,x1:x2] = arr_in
    return arr_out
    
    
