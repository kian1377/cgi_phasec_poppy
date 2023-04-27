import numpy as np
try:
    import cupy as cp
    cp.cuda.Device(0).compute_capability
except ImportError:
    pass

import poppy
from poppy.poppy_core import PlaneType

xp = cp if poppy.accel_math._USE_CUPY else np

from IPython.display import clear_output

from astropy.io import fits
import astropy.units as u
import time

import cgi_phasec_poppy
# from . import hlc, spc, polmap

import misc_funs as misc

import ray

class BBCGI():

    def __init__(self, 
                 actors, 
                 exp_time=None,
                 dm1_ref=np.zeros((48,48)),
                 dm2_ref=np.zeros((48,48)),
                ):
        
        self.Na = len(actors)
        self.actors = actors
        
        self.dm1_ref = dm1_ref
        self.dm2_ref = dm2_ref
        
        self.exp_time = exp_time
        
        self.psf_pixelscale = ray.get(actors[0].getattr.remote('psf_pixelscale'))
        self.psf_pixelscale_lamD = ray.get(actors[0].getattr.remote('psf_pixelscale_lamD'))
        
        self.Nact = 48
        self.npsf = ray.get(actors[0].getattr.remote('npsf'))
        
    def reset_dms(self):
        for i in range(self.Na):
            self.actors[i].set_dm1.remote(self.dm1_ref)
            self.actors[i].set_dm2.remote(self.dm2_ref)
    
    def flatten_dms(self):
        for i in range(self.Na):
            self.actors[i].set_dm1.remote(np.zeros((48,48)))
            self.actors[i].set_dm2.remote(np.zeros((48,48)))
        
    def set_dm1(self, dm_command):
        for i in range(self.Na):
            self.actors[i].set_dm1.remote(dm_command)
    
    def set_dm2(self, dm_command):
        for i in range(self.Na):
            self.actors[i].set_dm2.remote(dm_command)
        
    def add_dm1(self, dm_command):
        for i in range(self.Na):
            self.actors[i].add_dm1.remote(dm_command)
        
    def add_dm2(self, dm_command):
        for i in range(self.Na):
            self.actors[i].add_dm2.remote(dm_command)
        
    def get_dm1(self):
        return ray.get(self.actors[0].get_dm1.remote())
        
    def get_dm2(self):
        return ray.get(self.actors[0].get_dm2.remote())
    
    def use_fpm(self, val): # val is boolean true/false
        for i in range(self.Na):
            self.actors[i].setattr.remote('use_fpm', val)
        
    def calc_psfs(self, quiet=True):
        start = time.time()
        pending_psfs = []
        for i in range(self.Na):
            future_psfs = self.actors[i].calc_psf.remote()
            pending_psfs.append(future_psfs)
        psfs = xp.array(ray.get(pending_psfs))
        clear_output(wait=True)
        if not quiet: print('PSFs calculated in {:.3f}s.'.format(time.time()-start))
        return psfs
    
    def snap(self, EMCCD=None):
        pending_ims = []
        for i in range(self.Na):
            future_ims = self.actors[i].snap.remote()
            pending_ims.append(future_ims)
        ims = xp.array(ray.get(pending_ims))

        im = xp.sum(ims, axis=0)/self.Na # average each of the 

        clear_output(wait=True)
        
        if EMCCD is not None and self.exp_time is not None:
            im = EMCCD.sim_sub_frame(im, self.exp_time.to_value(u.s))
        
        return im

