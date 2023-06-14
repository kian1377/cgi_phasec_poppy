#   Copyright 2019 California Institute of Technology
# ------------------------------------------------------------------

# experience has shown that with the Intel math libraries the program
# can hang if run in a multithreaded condition with more than one 
# thread allocated, so set to 1. This must be done before importing numpy
import os
os.environ['MKL_NUM_THREADS'] = '1' 
import numpy as np
from .math_module import xp
# import proper

def ffts( wavefront, direction ):
    if wavefront.dtype != 'complex128' and wavefront.dtype != 'complex64':
        wavefront = wavefront.astype(complex)

    n = wavefront.shape[0]  # assumed to be power of 2
    wavefront[:,:] = xp.roll( xp.roll(wavefront, -n//2, 0), -n//2, 1 )  # shift to corner
    
    if direction == -1:
        wavefront[:,:] = xp.fft.fft2(wavefront) / xp.size(wavefront)
    else:
        wavefront[:,:] = xp.fft.ifft2(wavefront) * xp.size(wavefront)
    
    wavefront[:,:] = xp.roll( xp.roll(wavefront, n//2, 0), n//2, 1 )    # shift to center 

    return wavefront