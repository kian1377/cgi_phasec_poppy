import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
import astropy.units as u
import time
import ray

import cgi_phasec_poppy
from . import imshows, math_module
# from .math_module import xp, ensure_np_array

xp = math_module.xp

class ParallelizedCGI():

    def __init__(self, 
                 actors, 
                 exp_time=None,
                 EMCCD=None,
                 dm1_ref=np.zeros((48,48)),
                 dm2_ref=np.zeros((48,48)),
                 use_noise=False,
                 Iref=None,
                ):
        
        self.Na = len(actors) # total number of actors given
        self.actors = actors
        
        self.dm1_ref = dm1_ref
        self.dm2_ref = dm2_ref
        
        self.exp_time = exp_time
        self.EMCCD = EMCCD
        
        self.psf_pixelscale = ray.get(actors[0].getattr.remote('psf_pixelscale'))
        self.psf_pixelscale_lamD = ray.get(actors[0].getattr.remote('psf_pixelscale_lamD'))
        
        self.Nact = ray.get(actors[0].getattr.remote('Nact'))
        self.npsf = ray.get(actors[0].getattr.remote('npsf'))
        
        self.dm_mask = ray.get(actors[0].getattr.remote('dm_mask'))
        
        self.use_noise = use_noise
        
        self.Iref = Iref
    
    def set_actor_attr(self, attr, val):
        for i in range(self.Na):
            self.actors[i].setattr.remote(attr,val)
            
    
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
    
    def source_offset(self, val):
        for i in range(self.Na):
            self.actors[i].setattr.remote('source_offset', val)
    
    def calc_psfs(self, quiet=True):
        start = time.time()
        pending_psfs = []
        for i in range(self.Na):
            future_psfs = self.actors[i].calc_psf.remote()
            pending_psfs.append(future_psfs)
        psfs = ray.get(pending_psfs)
        if isinstance(psfs[0], np.ndarray):
            xp = np
        elif isinstance(psfs[0], cp.ndarray):
            xp = cp
        psfs = xp.array(psfs)
        
        if not quiet: print('PSFs calculated in {:.3f}s.'.format(time.time()-start))
        return psfs
    
    def snap(self):
        pending_ims = []
        for i in range(self.Na):
            future_ims = self.actors[i].snap.remote()
            pending_ims.append(future_ims)
        ims = ray.get(pending_ims)
        ims = xp.array(ims)
        
#         if self.EMCCD is not None and self.exp_time is not None:
#             im = xp.array(self.EMCCD.sim_sub_frame(ensure_np_array(im), self.exp_time.to_value(u.s)))
        
        if self.use_noise:
            im = self.add_noise(ims)
        else:
            im = xp.sum(ims, axis=0)/self.Na # average each of the images
            
        if self.Iref is not None:
            im /= self.Iref
            
        return im
    
    def add_noise(self, mono_images):
        
        if self.exp_time is None or self.dark_current_rate is None or self.read_noise is None:
            raise Exception('Must provide noise statistic values in order to add noise to an image.')
        else:
            exp_time = self.exp_time.to_value(u.second)
            dark_current_rate = self.dark_current_rate.to_value(u.electron/u.pix/u.s)
            read_noise = self.read_noise.to_value(u.electron/u.pix) # per frame but this code only supports 1 frame
            
        # add photon noise to each individual subframe then add read noise and dark current
        image_in_counts = xp.sum(mono_images, axis=0) * exp_time
        noisy_image_in_counts = xp.random.poisson(image_in_counts)
        noisy_image_in_electrons = self.gain * noisy_image_in_counts
        
        # Compute dark current
        darkCurrent = dark_current_rate * exp_time * xp.ones_like(image_in_counts)
        darkCurrent = xp.random.poisson(darkCurrent)

        # Compute Gaussian read noise
        readNoise = read_noise * xp.random.randn(image_in_counts.shape[0], image_in_counts.shape[1])

        # Convert back from e- to counts and then discretize
        noisy_image = xp.round( (self.gain*noisy_image_in_electrons + darkCurrent + readNoise) )
        
        return noisy_image