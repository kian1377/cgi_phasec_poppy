import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
import astropy.units as u
import time

import poppy
from poppy.poppy_core import PlaneType

import cgi_phasec_poppy
from . import hlc, hlc_dev, spc, polmap, math_module
from .math_module import xp, ensure_np_array
from . import utils
from .imshows import *
if poppy.accel_math._USE_CUPY:
    import cupy as cp
    math_module.update_xp(cp)

class CGI():

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
                 dm_inf_fun=None,
                 polaxis=0,
                 det_rotation=0, 
                 source_flux=None,
                 use_noise=False,
                 exp_time=None,
                 exp_time_ref=None,
                 gain=None,
                 gain_ref=None,
                 emccd=None,
                 use_shot_noise=False, 
                 Nframes=1,
                 Imax_ref=None,
                ):
        
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
        elif self.cgi_mode=='spc-wide' or cgi_mode=='spc-wide_band4' or cgi_mode=='spc-wide_band1':
            if 'band1' in cgi_mode:
                self.wavelength_c = 575e-9*u.m
            else:
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
        if psf_pixelscale_lamD is not None: # overrides psf_pixelscale this way
            self.psf_pixelscale_lamD = psf_pixelscale_lamD
            self.psf_pixelscale = 13e-6*u.m/u.pix / (0.5e-6/self.wavelength_c.value) * self.psf_pixelscale_lamD/0.5
        else:
            self.psf_pixelscale = psf_pixelscale
            self.psf_pixelscale_lamD = 1/2 * 0.5e-6/self.wavelength_c.value * self.psf_pixelscale.to(u.m/u.pix).value/13e-6
        self.interp_order = interp_order # interpolation order for resampling wavefront at detector
        
        self.init_mode_optics()

        self.dm_inf_fun = cgi_phasec_poppy.data_dir/'dm-acts'/'proper_inf_func.fits' if dm_inf_fun is None else dm_inf_fun
        self.init_dms()
        self.dm1_ref = dm1_ref
        self.dm2_ref = dm2_ref
        self.set_dm1(dm1_ref)
        self.set_dm2(dm2_ref)
        
        if self.use_opds: 
            self.init_opds()
        
        self.det_rotation = det_rotation

        self.optics = ['pupil', 'polmap', 'primary', 'secondary', 'poma_fold', 'm3', 'm4', 'm5', 'tt_fold', 'fsm', 'oap1', 
                       'focm', 'oap2', 'dm1', 'dm2', 'oap3', 'fold3', 'oap4', 'pupilmask', 'oap5', 'fpm', 'oap6',
                       'lyotstop', 'oap7', 'fieldstop', 'oap8', 'filter', 
                       'imaging_lens_lens1', 'imaging_lens_lens2', 'fold4', 'image']
        
        self.gain = gain
        self.use_noise = use_noise
        self.source_flux = source_flux
        self.emccd = emccd
        self.Nframes = Nframes
        self.use_shot_noise = use_shot_noise
        self.normalize = 'first' if source_flux is None else 'none'
        self.exp_time = exp_time
        self.gain_ref = gain_ref
        self.exp_time_ref = exp_time_ref
        self.Imax_ref = Imax_ref
        
           
    def init_mode_optics(self):
        self.FPM_plane = poppy.ScalarTransmission('FPM Plane (No Optic)', planetype=PlaneType.intermediate) # placeholder
        
        if self.cgi_mode=='hlc':
            self.optics_dir = cgi_phasec_poppy.data_dir/'hlc'
            
            self.PUPIL = poppy.FITSOpticalElement('Roman Pupil', 
                                                  transmission=str(self.optics_dir/'pupil_n310_new.fits'),
                                                  pixelscale=self.pupil_diam.value/310,
                                                  planetype=PlaneType.pupil)
    
            self.SPM = poppy.ScalarTransmission('SPM Plane (No Optic)', planetype=PlaneType.pupil)
        
            self.dm2_mask = poppy.FITSOpticalElement('DM2 Mask', 
                                                     transmission=str(self.optics_dir/'dm2mask.fits'),
                                                     pixelscale=0.00014935510078279392, 
                                                     planetype=PlaneType.intermediate)
            if self.use_fpm:
                # Find nearest available FPM wavelength that matches specified wavelength and initialize the FPM data
                lam_um = self.wavelength.value * 1e6
                f = open( str(self.optics_dir/'fpm_files.txt') )
                fpm_nlam = int(f.readline())
                fpm_lams = np.zeros((fpm_nlam),dtype=float)
                for j in range(0,fpm_nlam): 
                    fpm_lams[j] = float(f.readline())*1e-6
                fpm_root_fnames = [j.strip() for j in f.readlines()] 
                f.close()

                diff = np.abs(fpm_lams - self.wavelength.value)
                w = np.argmin( diff )
                if diff[w] > 0.1e-9: 
                    raise Exception('Only wavelengths within 0.1nm of avalable FPM wavelengths can be used.'
                                    'Closest available to requested wavelength is {}.'.format(fpm_lams[w]))
                fpm_rootname = self.optics_dir/fpm_root_fnames[w]

                fpm_r_fname = str(fpm_rootname)+'real.fits'
                fpm_i_fname = str(fpm_rootname)+'imag.fits'

                fpm_r = fits.getdata(fpm_r_fname)
                fpm_i = fits.getdata(fpm_i_fname)
                    
                self.fpm_phasor = fpm_r + 1j*fpm_i
                
#                 self.fpm_mask = (fpm_r != fpm_r[0,0]).astype(int)
                self.fpm_mask = (fpm_r != fpm_r[fpm_r.shape[0]-1,fpm_r.shape[0]-1]).astype(int)
                self.fpm_ref_wavelength = fits.getheader(fpm_r_fname)['WAVELENC']
                self.fpm_pixelscale_lamD = fits.getheader(fpm_r_fname)['PIXSCLLD']
            else:
                self.FPM = None
                
            self.LS = poppy.FITSOpticalElement('Lyot Stop', 
                                               transmission=str(self.optics_dir/'lyot_hlc_n310_new.fits'), 
                                               pixelscale=5.50105901118828e-05 * 309/310,
                                               planetype=PlaneType.pupil)
            
            if self.use_fieldstop: 
                radius = 9.8/(310/(self.npix*self.oversample)) * (self.wavelength_c/self.wavelength) * 7.229503001768824e-06*u.m
                self.fieldstop = poppy.CircularAperture(radius=radius, name='HLC Field Stop', gray_pixel=True)
            else: 
                self.fieldstop = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='Field Stop Plane (No Optic)')
                
        elif self.cgi_mode=='spc-spec': 
            self.optics_dir = cgi_phasec_poppy.data_dir/'spc-spec'            
            self.PUPIL = poppy.FITSOpticalElement('Roman Pupil', 
                                                  transmission=str(self.optics_dir/'pupil_SPC-20200617_1000.fits'),
                                                  planetype=PlaneType.pupil)
            self.SPM = poppy.FITSOpticalElement('SPM', 
                                                transmission=str(self.optics_dir/'SPM_SPC-20200617_1000_rounded9_rotated.fits'),
                                                planetype=PlaneType.pupil)
            self.LS = poppy.FITSOpticalElement('Lyot Stop',
                                               transmission=str(self.optics_dir/'LS_SPC-20200617_1000.fits'), 
                                               planetype=PlaneType.pupil)
            if self.use_fpm: 
                self.FPM = poppy.FixedSamplingImagePlaneElement('FPM', 
                                                                transmission=str(self.optics_dir/'fpm_0.05lamD.fits'))
            else: 
                self.FPM = poppy.ScalarTransmission(name='FPM Plane (No Optic)', planetype=PlaneType.intermediate) 
            self.fieldstop = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='Field Stop Plane (No Optic)')
            self.use_fieldstop = False
            
        elif 'spc-wide' in self.cgi_mode:
            self.optics_dir = cgi_phasec_poppy.data_dir/'spc-wide'            
            self.PUPIL = poppy.FITSOpticalElement('Roman Pupil', 
                                                  transmission=str(self.optics_dir/'pupil_SPC-20200610_1000.fits'),
                                                  planetype=PlaneType.pupil)
            self.SPM = poppy.FITSOpticalElement('SPM', 
                                                str(self.optics_dir/'SPM_SPC-20200610_1000_rounded9_gray_rotated.fits'),
                                                planetype=PlaneType.pupil)
            if self.use_fpm: 
                fpm_sampling_lamD = 0.1 * self.wavelength_c / self.wavelength
                fpm_sampling_arcsec = fpm_sampling_lamD * self.as_per_lamD/u.pix
                print(fpm_sampling_lamD, fpm_sampling_arcsec)
                self.FPM = poppy.FixedSamplingImagePlaneElement('FPM', 
                                                                str(self.optics_dir/'FPM_SPC-20200610_0.1_lamc_div_D.fits'),
                                                                wavelength_c=self.wavelength_c,
                                                                pixelscale=fpm_sampling_arcsec, 
                                                               )
            else: 
                self.FPM = poppy.ScalarTransmission(name='FPM Plane (No Optic)', planetype=PlaneType.intermediate) 
            
            self.LS = poppy.FITSOpticalElement('Lyot Stop',
                                               transmission=str(self.optics_dir/'LS_SPC-20200610_1000.fits'), 
                                               planetype=PlaneType.pupil)
            self.fieldstop = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='Field Stop Plane (No Optic)')
            self.use_fieldstop = False
        
        if self.polaxis!=0:
            polfile = cgi_phasec_poppy.data_dir/'pol'/'phasec_pol'
            polmap_amp, polmap_opd = polmap.polmap( str(polfile), 
                                                   self.wavelength, 
                                                   self.npix, int(self.oversample*self.npix), self.polaxis )
            self.POLMAP = poppy.ArrayOpticalElement(name='Polarization Error Map', 
                                                    opd=polmap_opd, transmission=polmap_amp,
                                                    pixelscale=self.pupil_diam/(self.npix*u.pix))
        else:
            self.POLMAP = poppy.ScalarTransmission('No Polarization Error Map')
            
        self.detector = poppy.Detector(pixelscale=self.psf_pixelscale, fov_pixels=self.npsf, interp_order=self.interp_order)
    
    def getattr(self, attr):
        return getattr(self, attr)
    
    def setattr(self, attr, val):
        setattr(self, attr, val)
        
    # DM methods
    def init_dms(self):
        self.Nact = 48
        self.dm_diam = 46.3*u.mm
        self.act_spacing = 0.9906*u.mm
        
        self.dm_mask = np.ones((self.Nact,self.Nact))
        xx = (np.linspace(0, self.Nact-1, self.Nact) - self.Nact/2 + 1/2) * self.act_spacing.to(u.mm).value*2
        x,y = np.meshgrid(xx,xx)
        r = np.sqrt(x**2 + y**2)
        self.dm_mask[r>47.5] = 0
        self.Nacts = int(self.dm_mask.sum())
        
        if poppy.accel_math._USE_CUPY:
            self.dm_zernikes = poppy.zernike.arbitrary_basis(cp.array(self.dm_mask), nterms=15, outside=0).get()
        else:
            self.dm_zernikes = poppy.zernike.arbitrary_basis(self.dm_mask, nterms=15, outside=0)
        
        self.DM1 = poppy.ContinuousDeformableMirror(dm_shape=(self.Nact,self.Nact), name='DM1', 
                                                    actuator_spacing=self.act_spacing, 
                                                    inclination_x=0, inclination_y=9.65,
                                                    influence_func=str(self.dm_inf_fun),
                                                    include_factor_of_two=True,
                                                    )
        self.DM2 = poppy.ContinuousDeformableMirror(dm_shape=(self.Nact,self.Nact), name='DM2', 
                                                    actuator_spacing=self.act_spacing, 
                                                    inclination_x=0, inclination_y=9.65,
                                                    influence_func=str(self.dm_inf_fun),
                                                    # radius=self.dm_diam/2,
                                                    include_factor_of_two=True,
                                                    )
        
    
    def reset_dms(self):
        self.set_dm1(self.dm1_ref)
        self.set_dm2(self.dm2_ref)
    
    def flatten_dms(self):
        self.set_dm1( np.zeros((self.Nact, self.Nact)) )
        self.set_dm2( np.zeros((self.Nact, self.Nact)) )
        
    def set_dm1(self, dm_command):
        self.DM1.set_surface(dm_command)
    
    def set_dm2(self, dm_command):
        self.DM2.set_surface(dm_command)
        
    def add_dm1(self, dm_command):
        self.DM1.set_surface(self.get_dm1() + dm_command)
        
    def add_dm2(self, dm_command):
        self.DM2.set_surface(self.get_dm2() + dm_command)
        
    def get_dm1(self):
        return self.DM1.surface.get()
        
    def get_dm2(self):
        return self.DM2.surface.get()
    
    def show_dms(self):
        wf = poppy.FresnelWavefront(beam_radius=self.dm_diam/2, npix=self.npix, oversample=1)
        misc.imshow2(self.get_dm1(), self.get_dm2(), 'DM1 Commands', 'DM2 Commands')
        misc.imshow2(self.DM1.get_opd(wf), self.DM2.get_opd(wf), 
                     'DM1 OPD', 'DM2 OPD', pxscl=self.dm_diam/self.npix)
    
    # utility functions
    def glass_index(self, glass):
        a = np.loadtxt( str( cgi_phasec_poppy.data_dir/'glass'/(glass+'_index.txt') ) )  # lambda_um index pairs
        f = interp1d( a[:,0], a[:,1], kind='cubic' )
        return f( self.wavelength.value*1e6 )
    
    def init_opds(self):
        opddir = cgi_phasec_poppy.data_dir/'opd-maps'

        opdunits = 'meters'

        self.primary_opd = poppy.FITSOpticalElement('Primary OPD', 
                                                    opd=str(opddir/'roman_phasec_PRIMARY_synthetic_phase_error_V1.0.fits'),
                                               opdunits=opdunits, planetype=PlaneType.intermediate)

        self.secondary_opd = poppy.FITSOpticalElement('Secondary OPD',
                                                 opd=str(opddir/'roman_phasec_SECONDARY_synthetic_phase_error_V1.0.fits'),
                                                 opdunits=opdunits, planetype=PlaneType.intermediate)

        self.poma_fold_opd = poppy.FITSOpticalElement('POMA-Fold OPD', 
                                                 opd=str(opddir/'roman_phasec_POMAFOLD_measured_phase_error_V1.1.fits'),
                                                 opdunits=opdunits,planetype=PlaneType.intermediate)

        self.m3_opd = poppy.FITSOpticalElement('M3 OPD', 
                                          opd=str(opddir/'roman_phasec_M3_measured_phase_error_V1.1.fits'), 
                                          opdunits=opdunits, planetype=PlaneType.intermediate)

        self.m4_opd = poppy.FITSOpticalElement('M4 OPD',
                                          opd=str(opddir/'roman_phasec_M4_measured_phase_error_V1.1.fits'), 
                                          opdunits=opdunits, planetype=PlaneType.intermediate)

        self.m5_opd = poppy.FITSOpticalElement('M5 OPD',
                                          opd=str(opddir/'roman_phasec_M5_measured_phase_error_V1.1.fits'),
                                          opdunits=opdunits, planetype=PlaneType.intermediate)

        self.tt_fold_opd = poppy.FITSOpticalElement('TT-Fold OPD', 
                                               opd=str(opddir/'roman_phasec_TTFOLD_measured_phase_error_V1.1.fits'), 
                                               opdunits=opdunits, planetype=PlaneType.intermediate)

        self.fsm_opd = poppy.FITSOpticalElement('FSM OPD', 
                                           opd=str(opddir/'roman_phasec_LOWORDER_phase_error_V2.0.fits'),
                                           opdunits=opdunits, planetype=PlaneType.intermediate)

        self.oap1_opd = poppy.FITSOpticalElement('OAP1 OPD',
                                            opd=str(opddir/'roman_phasec_OAP1_phase_error_V3.0.fits'),
                                            opdunits=opdunits, planetype=PlaneType.intermediate)

        self.focm_opd = poppy.FITSOpticalElement('FOCM OPD', 
                                            opd=str(opddir/'roman_phasec_FCM_EDU_measured_coated_phase_error_V2.0.fits'), 
                                            opdunits=opdunits, planetype=PlaneType.intermediate)

        self.oap2_opd = poppy.FITSOpticalElement('OAP2 OPD', 
                                            opd=str(opddir/'roman_phasec_OAP2_phase_error_V3.0.fits'), 
                                            opdunits=opdunits, planetype=PlaneType.intermediate)

        self.dm1_opd = poppy.FITSOpticalElement('DM1 OPD',
                                           opd=str(opddir/'roman_phasec_DM1_phase_error_V1.0.fits'), 
                                           opdunits=opdunits, planetype=PlaneType.intermediate)

        self.dm2_opd = poppy.FITSOpticalElement('DM2 OPD', 
                                           opd=str(opddir/'roman_phasec_DM2_phase_error_V1.0.fits'),
                                           opdunits=opdunits, planetype=PlaneType.intermediate)

        self.oap3_opd = poppy.FITSOpticalElement('OAP3 OPD',
                                            opd=str(opddir/'roman_phasec_OAP3_phase_error_V3.0.fits'), 
                                            opdunits=opdunits, planetype=PlaneType.intermediate)

        self.fold3_opd = poppy.FITSOpticalElement('F3 OPD',
                                             opd=str(opddir/'roman_phasec_FOLD3_FLIGHT_measured_coated_phase_error_V2.0.fits'), 
                                             opdunits=opdunits, planetype=PlaneType.intermediate)

        self.oap4_opd = poppy.FITSOpticalElement('OAP4 OPD', 
                                            opd=str(opddir/'roman_phasec_OAP4_phase_error_V3.0.fits'), 
                                            opdunits=opdunits, planetype=PlaneType.intermediate)

        self.pupil_fold_opd = poppy.FITSOpticalElement('SPM Fold OPD', 
                                                  opd=str(opddir/'roman_phasec_PUPILFOLD_phase_error_V1.0.fits'), 
                                                  opdunits=opdunits, planetype=PlaneType.intermediate)

        self.pupil_mask_opd = poppy.FITSOpticalElement('SPM OPD', 
                                                  opd=str(opddir/'roman_phasec_PUPILMASK_phase_error_V1.0.fits'), 
                                                  opdunits=opdunits, planetype=PlaneType.intermediate)

        self.oap5_opd = poppy.FITSOpticalElement('OAP5 OPD',
                                            opd=str(opddir/'roman_phasec_OAP5_phase_error_V3.0.fits'), 
                                            opdunits=opdunits, planetype=PlaneType.intermediate)

        self.oap6_opd = poppy.FITSOpticalElement('OAP6 OPD', 
                                            opd=str(opddir/'roman_phasec_OAP6_phase_error_V3.0.fits'), 
                                            opdunits=opdunits, planetype=PlaneType.intermediate)

        self.oap7_opd = poppy.FITSOpticalElement('OAP7 OPD', 
                                            opd=str(opddir/'roman_phasec_OAP7_phase_error_V3.0.fits'),
                                            opdunits=opdunits, planetype=PlaneType.intermediate)

        self.oap8_opd = poppy.FITSOpticalElement('OAP8 OPD', 
                                            opd=str(opddir/'roman_phasec_OAP8_phase_error_V3.0.fits'), 
                                            opdunits=opdunits, planetype=PlaneType.intermediate)

        self.filter_opd = poppy.FITSOpticalElement('Filter OPD',
                                              opd=str(opddir/'roman_phasec_FILTER_phase_error_V1.0.fits'), 
                                              opdunits=opdunits, planetype=PlaneType.intermediate)

        self.lens_opd = poppy.FITSOpticalElement('LENS OPD', 
                                            opd=str(opddir/'roman_phasec_LENS_phase_error_V1.0.fits'), 
                                            opdunits=opdunits, planetype=PlaneType.intermediate)
        
        
    def init_inwave(self):
        inwave = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength,
                                        npix=self.npix, oversample=self.oversample)
        
        if self.source_offset[0]>0 or self.source_offset[1]>0:
            inwave.tilt(Xangle=-self.source_offset[0]*self.as_per_lamD, Yangle=-self.source_offset[1]*self.as_per_lamD)
            
        if self.source_flux is not None:
            # scale input wavefront amplitude by the photon flux of source
            flux_per_pixel = self.source_flux * (inwave.pixelscale*u.pix)**2
            inwave.wavefront *= np.sqrt((flux_per_pixel).value)
            self.normalize = 'none'
        
        self.inwave = inwave
        
    def show_polmap(self):
        misc.imshow2(self.POLMAP.amplitude, self.POLMAP.opd,
                     'POLMAP: Amplitude', 'POLMAP: OPD', 
                     npix=self.npix, pxscl=self.POLMAP.pixelscale)
        
    def calc_wfs(self, quiet=False): # returns all poppy.FresnelWavefront objects for each plane
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
            
        self.init_inwave()
        if self.cgi_mode=='hlc':
            wfs = hlc.run(self, return_intermediates=True)
        else:
            wfs = spc.run(self, return_intermediates=True)
            
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
            
        return wfs
    
    def calc_psf(self, quiet=True): # returns just the poppy.FresnelWavefront object of the image plane
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
            
        self.init_inwave()
        if self.cgi_mode=='hlc':
            wfs = hlc.run(self, return_intermediates=False)
        else:
            wfs = spc.run(self, return_intermediates=False)
        
        psf_wf = wfs[-1].wavefront
        if self.Imax_ref is not None:
            psf_wf /= np.sqrt(self.Imax_ref)
        
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
        
        return psf_wf
    
    def snap(self): # returns just the intensity at the image plane
        if self.emccd is None:
            self.init_inwave()
            if self.cgi_mode=='hlc':
                wfs = hlc.run(self, return_intermediates=False)
            else:
                wfs = spc.run(self, return_intermediates=False)
            
            im = wfs[-1].intensity

            # if self.use_noise:
            #     im = self.add_noise(im)
            
            if self.Imax_ref is not None:
                im /= self.Imax_ref
                
            if self.exp_time is not None and self.exp_time_ref is not None:
                im /= (self.exp_time/self.exp_time_ref).value
                
            if self.gain is not None and self.gain_ref is not None:
                im /= self.gain/self.gain_ref
                
            return im
        
        elif self.emccd is not None:
    
            im = xp.abs(self.calc_psf())**2
        
            total_im = 0
            for i in range(self.Nframes):
                if self.source_flux is not None:
                    if self.use_shot_noise:
                        pass

                    im = self.emccd.sim_sub_frame(ensure_np_array(im), self.exp_time)

                total_im += im
            return total_im
    
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
    



        