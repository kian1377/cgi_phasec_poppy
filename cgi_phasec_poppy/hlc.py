import poppy
from poppy.poppy_core import PlaneType
from poppy import accel_math

try:
    import cupy as cp
except:
    pass

import numpy as np
import astropy.units as u
import copy

def run(HLC, return_intermediates=False):    
    # Define various optic focal lengths, diameters, and distances between optics.
    fl_pri = 2.838279206904720*u.m
    sm_despace_m = 0*u.m
    d_pri_sec = 2.285150508110035*u.m + sm_despace_m
    fl_sec = -0.654200796568004*u.m
    diam_sec = 0.58166*u.m
    d_sec_pomafold = 2.993753469304728*u.m + sm_despace_m
    diam_pomafold = 0.09*u.m
    d_pomafold_m3 = 1.680935841598811*u.m
    fl_m3 = 0.430216463069001*u.m
    diam_m3 = 0.2*u.m
    d_m3_pupil = 0.469156807765176*u.m
    d_m3_m4 = 0.943514749358944*u.m
    fl_m4 = 0.116239114833590*u.m
    diam_m4 = 0.07*u.m
    d_m4_m5 = 0.429145636743193*u.m
    fl_m5 = 0.198821518772608*u.m
    d_m5_pupil = 0.716529242927776*u.m
    diam_m5 = 0.07*u.m
    d_m5_ttfold = 0.351125431220770*u.m
    diam_ttfold = 0.06*u.m
    d_ttfold_fsm = d_m5_pupil - d_m5_ttfold 
    if HLC.use_pupil_defocus: d_ttfold_fsm = d_ttfold_fsm + 0.033609*u.m  # 33.6 mm to put pupil 6 mm from SPC mask
    diam_fsm = 0.0508*u.m
    d_fsm_oap1 = 0.354826767220001*u.m
    fl_oap1 = 0.503331895563883*u.m
    diam_oap1 = 0.060*u.m
    focm_z_shift_m = 0*u.m
    d_oap1_focm = 0.768029932093727*u.m + focm_z_shift_m
    diam_focm = 0.035*u.m
    d_focm_oap2 = 0.314507535543064*u.m + focm_z_shift_m
    fl_oap2 = 0.579205571254990*u.m
    diam_oap2 = 0.060*u.m
    d_oap2_dm1 = 0.775857408587825*u.m
    d_dm1_dm2 = 1.0*u.m
    d_dm2_oap3 = 0.394833855161549*u.m
    fl_oap3 = 1.217276467668519*u.m
    diam_oap3 = 0.06*u.m
    d_oap3_fold3 = 0.505329955078121*u.m
    diam_fold3 = 0.06*u.m
    d_fold3_oap4 = 1.158897671642761*u.m
    fl_oap4 = 0.446951159052363*u.m
    diam_oap4 = 0.06*u.m
    d_oap4_pupilmask = 0.423013568764728*u.m
    d_pupilmask_oap5 = 0.408810704327559*u.m
    fl_oap5 = 0.548189354706822*u.m
    diam_oap5 = 0.06*u.m
    d_oap5_fpm = fl_oap5                    # to front of FPM 
    fpm_thickness = 0.006363747896388863*u.m    # account for FPM thickness (inclination included)
    fpm_index = HLC.glass_index('SILICA')
    d_fpm_oap6 = fpm_thickness / fpm_index + 0.543766629917668*u.m     # from front of FPM
    fl_oap6 = d_fpm_oap6
    diam_oap6 = 0.054*u.m
    d_oap6_lyotstop = 0.687476361491529*u.m
    d_oap6_exitpupil = d_oap6_lyotstop - 6e-3*u.m
    d_lyotstop_oap7 = 0.401748561745987*u.m
    fl_oap7 = 0.708251420923810*u.m
    diam_oap7 = 0.054*u.m
    d_oap7_fieldstop = fl_oap7 
    d_fieldstop_oap8 = 0.210985170345932*u.m * 0.997651
    fl_oap8 = d_fieldstop_oap8
    diam_oap8 = 0.026*u.m
    d_oap8_pupil = 0.237561587674008*u.m
    d_pupil_filter = 0.130*u.m
    d_oap8_filter = d_oap8_pupil + d_pupil_filter   # to front of filter
    diam_filter = 0.009*u.m
    filter_thickness = 0.004016105782012525*u.m      # account for filter thickness (inclination included)
    filter_index = HLC.glass_index('SILICA')
    d_filter_lens = filter_thickness / filter_index + 0.210581269256657095*u.m  # from front of filter
    diam_lens = 0.0104*u.m
    d_lens_fold4 = 0.202432155667761*u.m
#     if use_pupil_lens != 0: d_lens_fold4 = d_lens_fold4 - 0.0002*u.m # from back of pupil imaging lens
#     elif use_defocus_lens != 0: d_lens_fold4 = d_lens_fold4 + 0.001*u.m # doublet is 1 mm longer than singlet, so make up for it
    diam_fold4 = 0.036*u.m
    d_fold4_image = 0.050000152941020161*u.m
    
    # define parameters related to final imaging lens, which is an air-gapped doublet
    lens_1_index = HLC.glass_index('S-BSL7R')
    lens_2_index = HLC.glass_index('PBM2R')
    r11, r12, lens_1_t = (0.10792660718579995*u.m, -0.10792660718579995*u.m, 0.003*u.m)
    r21, r22, lens_2_t = (1e10*u.m, 0.10608379812011390*u.m, 0.0025*u.m)
    air_gap = 0.0005*u.m
    
    fl_1 = 1 / ( (lens_1_index - 1) * ( 1.0/r11 - 1.0/r12 + (lens_1_index - 1)*lens_1_t / (lens_1_index*r11*r12) ) )
    d_pp_11 = -fl_1*(lens_1_index - 1)*lens_1_t / (lens_1_index*r12)
    d_pp_12 = -fl_1*(lens_1_index - 1)*lens_1_t / (lens_1_index*r11)

    fl_2 = 1 / ( (lens_2_index - 1) * ( 1.0/r21 - 1.0/r22 + (lens_2_index - 1)*lens_2_t / (lens_2_index*r21*r22) ) )
    d_pp_21 = -fl_2*(lens_2_index - 1)*lens_2_t / (lens_2_index*r22)
    d_pp_22 = -fl_2*(lens_2_index - 1)*lens_2_t / (lens_2_index*r21)

    d_filter_lens_1_pp1 = d_filter_lens + d_pp_11 
    d_lens_1_pp2_lens_2_pp1 = -d_pp_12 + air_gap + d_pp_21
    d_lens_2_pp2_fold4 = -d_pp_22 + d_lens_fold4

    # Define optical elements
    primary = poppy.QuadraticLens(fl_pri, name='Primary')
    secondary = poppy.QuadraticLens(fl_sec, name='Secondary')
    poma_fold = poppy.CircularAperture(radius=diam_pomafold/2, name="POMA_Fold")
    m3 = poppy.QuadraticLens(fl_m3, name='M3')
    m4 = poppy.QuadraticLens(fl_m4, name='M4')
    m5 = poppy.QuadraticLens(fl_m5, name='M5')
    tt_fold = poppy.CircularAperture(radius=diam_ttfold/2,name="TT_Fold")
    fsm = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='FSM')
    oap1 = poppy.QuadraticLens(fl_oap1, name='OAP1')
    focm = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='FOCM')
    oap2 = poppy.QuadraticLens(fl_oap2, name='OAP2')
    oap3 = poppy.QuadraticLens(fl_oap3, name='OAP3')
    fold3 = poppy.CircularAperture(radius=diam_fold3/2,name="Fold3")
    oap4 = poppy.QuadraticLens(fl_oap4, name='OAP4')
    oap5 = poppy.QuadraticLens(fl_oap5, name='OAP5')
    oap6 = poppy.QuadraticLens(fl_oap6, name='OAP6')
    oap7 = poppy.QuadraticLens(fl_oap7, name='OAP7')
    oap8 = poppy.QuadraticLens(fl_oap8, name='OAP8')
    filt = poppy.CircularAperture(radius=diam_filter/2, name='Filter')
    lens_1 = poppy.QuadraticLens(fl_1, name='LENS 1') # first lens of the doublet
    lens_2 = poppy.QuadraticLens(fl_2, name='LENS 2')
    fold4 = poppy.CircularAperture(radius=diam_fold4/2,name="Fold4")
        
    # Create the first part of the optical system
    fosys1 = poppy.FresnelOpticalSystem(name='HLC Pre-FPM', pupil_diameter=HLC.pupil_diam, 
                                        npix=HLC.npix, beam_ratio=1/HLC.oversample, verbose=True)
    
    fosys1.add_optic(HLC.PUPIL)
    fosys1.add_optic(HLC.POLMAP)
    fosys1.add_optic(primary)
    if HLC.use_opds: fosys1.add_optic(HLC.primary_opd)
        
    fosys1.add_optic(secondary, distance=d_pri_sec)
    if HLC.use_opds: fosys1.add_optic(HLC.secondary_opd)
        
    fosys1.add_optic(poma_fold, distance=d_sec_pomafold)
    if HLC.use_opds: fosys1.add_optic(HLC.poma_fold_opd)
        
    fosys1.add_optic(m3, distance=d_pomafold_m3)
    if HLC.use_opds: fosys1.add_optic(HLC.m3_opd)
        
    fosys1.add_optic(m4, distance=d_m3_m4)
    if HLC.use_opds: fosys1.add_optic(HLC.m4_opd)
        
    fosys1.add_optic(m5, distance=d_m4_m5)
    if HLC.use_opds: fosys1.add_optic(HLC.m5_opd)
        
    fosys1.add_optic(tt_fold, distance=d_m5_ttfold)
    if HLC.use_opds: fosys1.add_optic(HLC.tt_fold_opd)
        
    fosys1.add_optic(fsm, distance=d_ttfold_fsm)
    if HLC.use_opds: fosys1.add_optic(HLC.fsm_opd)
        
    fosys1.add_optic(oap1, distance=d_fsm_oap1)
    if HLC.use_opds: fosys1.add_optic(HLC.oap1_opd)
        
    fosys1.add_optic(focm, distance=d_oap1_focm)
    if HLC.use_opds: fosys1.add_optic(HLC.focm_opd)
        
    fosys1.add_optic(oap2, distance=d_focm_oap2)
    if HLC.use_opds: fosys1.add_optic(HLC.oap2_opd)
        
    fosys1.add_optic(HLC.DM1, distance=d_oap2_dm1)
    if HLC.use_opds: fosys1.add_optic(HLC.dm1_opd)
        
    fosys1.add_optic(HLC.DM2, distance=d_dm1_dm2)
    if HLC.use_opds: fosys1.add_optic(HLC.dm2_opd)
#     fosys1.add_optic(HLC.dm2_mask)
    
    fosys1.add_optic(oap3, distance=d_dm2_oap3)
    if HLC.use_opds: fosys1.add_optic(HLC.oap3_opd)
        
    fosys1.add_optic(fold3, distance=d_oap3_fold3)
    if HLC.use_opds: fosys1.add_optic(HLC.fold3_opd)
        
    fosys1.add_optic(oap4, distance=d_fold3_oap4)
    if HLC.use_opds: fosys1.add_optic(HLC.oap4_opd)
        
    fosys1.add_optic(HLC.SPM, distance=d_oap4_pupilmask)
    if HLC.use_opds: fosys1.add_optic(HLC.pupil_fold_opd)
        
    fosys1.add_optic(oap5, distance=d_pupilmask_oap5)
    if HLC.use_opds: fosys1.add_optic(HLC.oap5_opd)
        
    fosys1.add_optic(HLC.FPM_plane, distance=d_oap5_fpm)
    
    # Create second part of the optical system
    fosys2 = poppy.FresnelOpticalSystem(name='HLC Post-FPM', npix=HLC.npix, beam_ratio=1/HLC.oversample, verbose=True)
    
    fosys2.add_optic(HLC.FPM_plane)
    fosys2.add_optic(oap6, distance=d_fpm_oap6)
    if HLC.use_opds: fosys2.add_optic(HLC.oap6_opd)
        
    fosys2.add_optic(HLC.LS, distance=d_oap6_lyotstop)
    
    fosys2.add_optic(oap7, distance=d_lyotstop_oap7)
    if HLC.use_opds: fosys2.add_optic(HLC.oap7_opd)
        
    fosys2.add_optic(HLC.fieldstop, distance=d_oap7_fieldstop)
    
    fosys2.add_optic(oap8, distance=d_fieldstop_oap8)
    if HLC.use_opds: fosys2.add_optic(HLC.oap8_opd)
        
    fosys2.add_optic(filt, distance=d_oap8_filter)
    if HLC.use_opds: fosys2.add_optic(HLC.filter_opd)
        
    fosys2.add_optic(lens_1, distance=d_filter_lens_1_pp1)
    if HLC.use_opds: fosys2.add_optic(HLC.lens_opd)
        
    fosys2.add_optic(lens_2, distance=d_lens_1_pp2_lens_2_pp1)
    
    fosys2.add_optic(fold4, distance=d_lens_2_pp2_fold4)

    fosys2.add_optic(HLC.detector, distance=d_fold4_image)

#     fosys2.add_optic(HLC.detector, distance=d_lens_2_pp2_fold4 + d_fold4_image)
    
    # Calculate a psf from the first optical system to retrieve the final wavefront at the FPM plane 
    fpm_hdu, wfs_to_fpm = fosys1.calc_psf(wavelength=HLC.wavelength, inwave=HLC.inwave, 
                                          return_final=True, return_intermediates=return_intermediates)
    inwave2 = copy.deepcopy(wfs_to_fpm[-1]) # copy Wavefront object for use in the post FPM system
    
    if HLC.use_fpm: 
        fpm_pxscl_lamD = ( HLC.fpm_pixelscale_lamD * HLC.wavelength_c.to(u.meter) / HLC.wavelength.to(u.meter) ).value
        nfpm = HLC.fpm_phasor.shape[0]
        n = inwave2.wavefront.shape[0]
        nfpmlamD = nfpm*fpm_pxscl_lamD*inwave2.oversample
        mft = poppy.matrixDFT.MatrixFourierTransform(centering='FFTSTYLE')

        # Apply the FPM with MFTs
        inwave2.wavefront = accel_math._ifftshift(inwave2.wavefront)
        inwave2.wavefront = accel_math.fft_2d(inwave2.wavefront, forward=False, fftshift=True) # do a forward FFT to virtual pupil
        inwave2.wavefront *= HLC.fpm_phasor[-1,-1]
        wavefront_fpm = mft.perform(inwave2.wavefront, nfpmlamD, nfpm) # MFT back to highly sampled focal plane
        
        if poppy.accel_math._USE_CUPY:
            wavefront_fpm *= cp.array(HLC.fpm_mask * (HLC.fpm_phasor - 1))
        else:
            wavefront_fpm *= HLC.fpm_mask * (HLC.fpm_phasor - 1)
            
        wavefront_fpm = mft.inverse(wavefront_fpm, nfpmlamD, n) # MFT to virtual pupil
        inwave2.wavefront += wavefront_fpm
        inwave2.wavefront = accel_math.fft_2d(inwave2.wavefront, forward=True, fftshift=True) # FFT back to normally sampled focus
        inwave2.wavefront = accel_math._fftshift(inwave2.wavefront)
        
    psf_hdu, wfs_from_fpm = fosys2.calc_psf(wavelength=HLC.wavelength, inwave=inwave2, normalize='none',
                                            return_final=True, return_intermediates=return_intermediates,)
    
    if return_intermediates:
        wfs_to_fpm.pop(-1)
        wfs = wfs_to_fpm + wfs_from_fpm
    else: 
        wfs = wfs_from_fpm

    return wfs

#     0.10490069216127479 0.000993583483544923 -0.000993583483544923
#     -0.17078291119000114 0.0015421040653161094 1.635922563452011e-14
#     0.21432764805243154 0.0030356875488610326 0.25243230860876475


import ray
runner = ray.remote(run)

