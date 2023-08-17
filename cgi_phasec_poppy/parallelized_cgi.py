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
                 dm1_ref=np.zeros((48,48)),
                 dm2_ref=np.zeros((48,48)),
                 use_noise=False,
                 exp_time=None,
                 exp_time_ref=None,
                 gain=1,
                 gain_ref=None,
                 Imax_ref=None,
                ):
        
        self.Na = len(actors) # total number of actors given
        self.actors = actors
        
        self.dm1_ref = dm1_ref
        self.dm2_ref = dm2_ref
        self.set_dm1(dm1_ref)
        self.set_dm2(dm2_ref)
        
        self.exp_time = exp_time
        self.gain = gain
        
        self.psf_pixelscale = ray.get(actors[0].getattr.remote('psf_pixelscale'))
        self.psf_pixelscale_lamD = ray.get(actors[0].getattr.remote('psf_pixelscale_lamD'))
        
        self.Nact = ray.get(actors[0].getattr.remote('Nact'))
        self.Nacts = ray.get(actors[0].getattr.remote('Nacts'))
        self.npsf = ray.get(actors[0].getattr.remote('npsf'))
        
        self.dm_mask = ray.get(actors[0].getattr.remote('dm_mask'))
        
        self.use_noise = use_noise
        
        self.gain_ref = None
        self.exp_time_ref = None
        self.Imax_ref = Imax_ref
    
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
        
        if self.use_noise:
            im = xp.sum(ims, axis=0)
            im = self.add_noise(im)
        elif ray.get(self.actors[0].getattr.remote('source_flux')) is not None:
            im = xp.sum(ims, axis=0)
        else:
            im = xp.sum(ims, axis=0)/self.Na
            
        if self.Imax_ref is not None:
            im /= self.Imax_ref
            
        if self.exp_time is not None and self.exp_time_ref is not None:
            im /= (self.exp_time/self.exp_time_ref).value
            
        if self.gain is not None and self.gain_ref is not None:
            im /= self.gain/self.gain_ref
            
        return im
    
    def add_noise(self, image):
        
        if self.exp_time is None or self.dark_current_rate is None or self.read_noise is None:
            raise Exception('Must provide noise statistic values in order to add noise to an image.')
        else:
            exp_time = self.exp_time.to_value(u.second)
            dark_current_rate = self.dark_current_rate.to_value(u.electron/u.pix/u.s)
            read_noise_std = self.read_noise.to_value(u.electron/u.pix) # per frame but this code only supports 1 frame
        
        image_in_counts = image * exp_time
        # Add photon shot noise
        noisy_image_in_counts = xp.random.poisson(image_in_counts)
        
        noisy_image_in_e = self.gain * noisy_image_in_counts

        # Compute dark current
        dark = dark_current_rate * exp_time * xp.ones_like(image)
        dark = xp.random.poisson(dark)

        # Compute Gaussian read noise
        read = read_noise_std * xp.random.randn(image.shape[0], image.shape[1])

        # Convert back from e- to counts and then discretize
        noisy_image = xp.round( (noisy_image_in_e + dark + read) )
        
        
        return noisy_image
    