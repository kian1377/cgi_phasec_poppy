import numpy as np
from astropy.io import fits
import astropy.units as u
import time
try:
    import proper
    import roman_phasec_proper
    roman_phasec_proper.copy_here()
except ImportError:
    print('PROPER CGI model not installed and is unavailable')

import cgi_phasec_poppy
from . import imshows, math_module
from .math_module import xp

def pad_or_crop( arr_in, npix ):
    n_arr_in = arr_in.shape[0]
    if n_arr_in == npix:
        return arr_in
    elif npix < n_arr_in:
        x1 = n_arr_in // 2 - npix // 2
        x2 = x1 + npix
        arr_out = arr_in[x1:x2,x1:x2].copy()
    else:
        arr_out = np.zeros((npix,npix), dtype=arr_in.dtype)
        x1 = npix // 2 - n_arr_in // 2
        x2 = x1 + n_arr_in
        arr_out[x1:x2,x1:x2] = arr_in
    return arr_out


class PROPERCGI():

    def __init__(self, 
                 cgi_mode='hlc', 
                 wavelength=None, 
                 source_offset=(0,0), 
                 npsf=64, psf_pixelscale=13e-6*u.m/u.pix, psf_pixelscale_lamD=None, interp_order=3,
                 use_fpm=True, 
                 use_fieldstop=True, 
                 use_pupil_defocus=True, 
                 use_opds=False, 
                 dm1_ref=np.zeros((48,48)),
                 dm2_ref=np.zeros((48,48)),
                 polaxis=0,
                 source_flux=None,
                 exp_time=None,
                ):
        math_module.update_xp(np)
        self.cgi_mode = cgi_mode
        
        self.pupil_diam = 2.363114*u.m
        if self.cgi_mode=='hlc': 
            self.wavelength_c = 575e-9*u.m
            self.npix = 310
            self.oversample = 1024/310
        elif self.cgi_mode=='spc-spec':
            self.wavelength_c = 730e-9*u.m
            self.npix = 1000
            self.oversample = 2048/1000
        elif self.cgi_mode=='spc-wide':
            self.wavelength_c = 825e-9*u.m
            self.npix = 1000
            self.oversample = 2048/1000
            
        self.as_per_lamD = ((self.wavelength_c/self.pupil_diam)*u.radian).to(u.arcsec)
        
        if wavelength is None: 
            self.wavelength = self.wavelength_c
        else: 
            self.wavelength = wavelength
        
        self.source_offset = source_offset
        self.use_fpm = use_fpm
        self.use_pupil_defocus = use_pupil_defocus
        self.use_fieldstop = use_fieldstop
        self.use_opds = use_opds
        self.polaxis = polaxis
        
        self.npsf = npsf
        self.npsf_prop = int(2**np.ceil(np.log2(self.npsf)))
        if psf_pixelscale_lamD is not None: # overrides psf_pixelscale this way
            self.psf_pixelscale_lamD = psf_pixelscale_lamD
            self.psf_pixelscale = 13e-6*u.m/u.pix / (0.5e-6/self.wavelength_c.value) * self.psf_pixelscale_lamD/0.5
        else:
            self.psf_pixelscale = psf_pixelscale
            self.psf_pixelscale_lamD = 1/2 * 0.5e-6/self.wavelength_c.value * self.psf_pixelscale.to(u.m/u.pix).value/13e-6
        
        self.Nact = 48
        self.dm_diam = 46.3*u.mm
        self.act_spacing = 0.9906*u.mm
        
        self.dm_mask = np.ones((self.Nact,self.Nact))
        xx = (np.linspace(0, self.Nact-1, self.Nact) - self.Nact/2 + 1/2) * self.act_spacing.to(u.mm).value*2
        x,y = np.meshgrid(xx,xx)
        r = np.sqrt(x**2 + y**2)
        self.dm_mask[r>47] = 0
        
        self.set_dm1(dm1_ref)
        self.set_dm2(dm2_ref)
        
    def getattr(self, attr):
        return getattr(self, attr)
    
    def setattr(self, attr, val):
        setattr(self, attr, val)
    
    def reset_dms(self):
        self.DM1 = self.dm1_ref
        self.DM2 = self.dm2_ref
    
    def flatten_dms(self):
        self.DM1 = np.zeros((self.Nact, self.Nact))
        self.DM2 = np.zeros((self.Nact, self.Nact))
        
    def set_dm1(self, dm_command):
        self.DM1 = dm_command
    
    def set_dm2(self, dm_command):
        self.DM2 = dm_command
        
    def add_dm1(self, dm_command):
        self.DM1 += dm_command
        
    def add_dm2(self, dm_command):
        self.DM2 += dm_command
        
    def get_dm1(self):
        return self.DM1
        
    def get_dm2(self):
        return self.DM2
    
#     def show_dms(self):
#         wf = poppy.FresnelWavefront(beam_radius=self.dm_diam/2, npix=self.npix, oversample=1)
#         misc.imshow2(self.get_dm1(), self.get_dm2(), 'DM1 Commands', 'DM2 Commands')
#         misc.imshow2(self.DM1.get_opd(wf), self.DM2.get_opd(wf), 
#                      'DM1 OPD', 'DM2 OPD', pxscl=self.dm_diam/self.npix)
    
#     def show_polmap(self):
#         misc.imshow2(self.POLMAP.amplitude, self.POLMAP.opd,
#                      'POLMAP: Amplitude', 'POLMAP: OPD', 
#                      npix=self.npix, pxscl=self.POLMAP.pixelscale)
    
    def calc_psf(self, quiet=True): # returns just the poppy.FresnelWavefront object of the image plane
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        
        options = {
            'cor_type':self.cgi_mode, # change coronagraph type to correct band
            'final_sampling_lam0':self.psf_pixelscale_lamD, 
            'source_x_offset':self.source_offset[0],
            'source_y_offset':self.source_offset[1],
            'use_fpm':self.use_fpm,
            'use_field_stop':self.use_fieldstop,
            'use_errors':self.use_opds,
            'use_lens_errors':self.use_opds,
            'use_pupil_defocus':self.use_pupil_defocus,
            'use_hlc_dm_patterns':0,
            'use_dm1':1, 'dm1_m':self.DM1, 
            'use_dm2':1, 'dm2_m':self.DM2,
            'polaxis':self.polaxis,   
        }
        
        wf, pxscl_m = proper.prop_run('roman_phasec', 
                                      self.wavelength.to_value(u.um), 
                                      int(2**np.ceil(np.log2(self.npsf))), # must be nearest power of 2
                                      PASSVALUE=options,
                                      QUIET=True)
        
        wf = pad_or_crop(wf, self.npsf)
        
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
            
        return wf
    
    def snap(self): # returns just the intensity at the image plane
        
        options = {
            'cor_type':self.cgi_mode, # change coronagraph type to correct band
            'final_sampling_lam0':self.psf_pixelscale_lamD, 
            'source_x_offset':self.source_offset[0],
            'source_y_offset':self.source_offset[1],
            'use_fpm':self.use_fpm,
            'use_field_stop':self.use_fieldstop,
            'use_errors':self.use_opds,
            'use_lens_errors':self.use_opds,
            'use_pupil_defocus':self.use_pupil_defocus,
            'use_hlc_dm_patterns':0,
            'use_dm1':1, 'dm1_m':self.DM1, 
            'use_dm2':1, 'dm2_m':self.DM2,
            'polaxis':self.polaxis,   
        }

        wf, pxscl_m = proper.prop_run('roman_phasec', 
                                      self.wavelength.to_value(u.um), 
                                      int(2**np.ceil(np.log2(self.npsf))), # must be nearest power of 2
                                      PASSVALUE=options,
                                      QUIET=True)
        
        wf = pad_or_crop(wf, self.npsf)
        
        return np.abs(wf)**2


        