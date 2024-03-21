import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
import astropy.units as u
import time
import ray
import copy

import cgi_phasec_poppy
from . import imshows
from .math_module import xp, ensure_np_array

# xp = math_module.xp

class ParallelizedCGI():

    def __init__(self, 
                 actors, 
                 dm1_ref=np.zeros((48,48)),
                 dm2_ref=np.zeros((48,48)),
                 use_noise=False,
                 use_photon_noise=False,
                 exp_time=None,
                 gain=1,
                 EMCCD=None,
                 normalize=False,
                 exp_time_ref=None,
                 em_gain_ref=None,
                 dark_frame=False,
                 Imax_ref=1,
                 exp_times_list=None, 
                 Nframes_list=None,
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
        self.use_photon_noise = use_photon_noise
        
        self.normalize = normalize
        self.em_gain_ref = em_gain_ref
        self.exp_time_ref = exp_time_ref
        self.Imax_ref = Imax_ref

        self.EMCCD = EMCCD
        self.dark_frame = dark_frame
        self.subtract_bias = False

        self.exp_times_list = exp_times_list
        self.gain_list = None
        self.Nframes_list = Nframes_list

    
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
        psfs = xp.array(psfs)
        
        if not quiet: print('PSFs calculated in {:.3f}s.'.format(time.time()-start))
        return psfs
    
    def calc_images(self, quiet=True):
        if not quiet: print('Computing each monochromatic image.')
        pending_ims = []
        for i in range(self.Na):
            future_ims = self.actors[i].snap.remote()
            pending_ims.append(future_ims)
        ims = ray.get(pending_ims)
        return xp.array(ims)

    def add_photon_noise(self, flux_im, exp_time): # flux_im in units of ph/s/pix
        photon_counts_im = flux_im*exp_time
        noisy_im = xp.random.poisson(photon_counts_im)
        noisy_flux_im = noisy_im.astype(xp.float64)/exp_time
        return noisy_flux_im

    def snap(self, quiet=True):
        ims = self.calc_images(quiet=quiet)
        bbim = xp.sum(ims, axis=0)
        
        if self.EMCCD is None and self.use_photon_noise:
            bbim *= self.exp_time
            noisy_im = xp.random.poisson(bbim)
            return noisy_im
        elif self.EMCCD is not None: 
            emccd_im = 0.0
            for i in range(self.Nframes):
                if not quiet: print(f'Simulating EMCCD frame {i+1:d}.')
                emccd_im += self.EMCCD.sim_sub_frame(ensure_np_array(bbim), self.exp_time)
            emccd_im = xp.array(emccd_im)/self.Nframes

            if self.normalize: 
                if not quiet: print('Normalizing to max value')
                self.norm_factor = 1/self.Imax_ref * self.exp_time_ref/self.exp_time * self.em_gain_ref/self.EMCCD.em_gain
                normalized_im = emccd_im * self.norm_factor
                return normalized_im
            return emccd_im
        else:
            return bbim/self.Imax_ref

    def snap_many(self, quiet=True, plot=False, sat_thresh=100):

        if self.EMCCD is None:
            raise ValueError('ERROR: must have EMCCD object to use this functionality.')
        if self.exp_times_list is None:
            raise ValueError('ERROR: must specify a list of exposure times to stack frames with variable exposures.')
        if self.gain_list is None:
            self.gain_list = [self.EMCCD.em_gain]*len(self.exp_times_list)

        mono_flux_ims = self.calc_images(quiet=quiet)
        bb_flux_im = xp.sum(mono_flux_ims, axis=0) # assuimng source flux pre-implemented
        if plot: imshows.imshow1(bb_flux_im, 'Total image flux [ph/s/pix]', lognorm=True)

        # ims = []
        # im_masks = []
        total_flux = 0.0
        pixel_weights = 0.0
        for i in range(len(self.exp_times_list)):
            if not quiet: print(f'Generating all frames based on computed flux for exposure time {i+1:d}/{len(self.exp_times_list):d}: {self.exp_times_list[i]:.3f}s')
            self.exp_time = copy.copy(self.exp_times_list[i])
            self.EMCCD.em_gain = self.gain_list[i]
            if self.Nframes_list is not None:
                self.Nframes = self.Nframes_list[i]

            averaged_frame = 0.0
            for j in range(self.Nframes):
                emccd_frame = self.EMCCD.sim_sub_frame(ensure_np_array(bb_flux_im), self.exp_time)
                if plot: imshows.imshow1(emccd_frame, f'Individual EMCCD frame::\nExposure Time = {self.exp_times_list[i]}s\nGain = {self.EMCCD.em_gain:.1f}', lognorm=True)
                averaged_frame += emccd_frame
            averaged_frame = xp.array(averaged_frame)/self.Nframes
            pixel_sat_mask = averaged_frame > (2**self.EMCCD.nbits - sat_thresh)

            if self.subtract_bias:
                averaged_frame -= self.EMCCD.bias
            
            pixel_weights += ~pixel_sat_mask
            flux_im = averaged_frame/self.exp_time/self.EMCCD.em_gain # normalize by the exposure time and gain for this frame
            flux_im[pixel_sat_mask] = 0 # mask out the saturated pixels

            if plot: 
                imshows.imshow3(pixel_sat_mask, averaged_frame, flux_im, 
                                'Pixel Saturation Mask', 
                                f'Averaged EMCCD Frame:\nExposure Time = {self.exp_times_list[i]}s\nGain = {self.EMCCD.em_gain:.1f}\nN-frames = {self.Nframes:d}', 
                                'Masked Flux Image', 
                                lognorm2=True, lognorm3=True)
                
            total_flux += flux_im
            
        total_flux_im = total_flux/pixel_weights

        return total_flux_im/self.Imax_ref
    
    def snap_dark(self):
        dark_frame = 0.0
        for i in range(self.Nframes):
            dark_frame += self.EMCCD.sim_sub_frame(np.zeros((self.npsf, self.npsf)), self.exp_time)
        return xp.array(dark_frame)/self.Nframes
    
