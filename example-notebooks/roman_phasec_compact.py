#   Copyright 2021 California Institute of Technology
# ------------------------------------------------------------------

# Version 0.9, 15 June 2020, John Krist; beta-test version for mask evaluation
# Version 0.9.1, 29 June 2020, John Krist; rotated SPC pupils & Lyot stops 
# Version 0.9.2, 7 July 2020, John Krist; updated HLC occulters
# Version 1.0, 1.0.1, 15 Sept 2020, John Krist: updated HLC
# Version 1.1, 3 Dec 2020, John Krist: added field-dependent phase term at image
#     plane to match E-field from full model due to pupil offset
# Version 1.2, 2 Feb 2021, John Krist: switched to using MFTs to propagate to/from HLC FPM;
#     updated the list of available HLC FPM wavelengths; removed field-dependent phase term 
#     (now subtracting it from the full model instead); read in list of available HLC FPM files;
#     added spc-mswc mode; added modes to allow spc-wide or spc-mswc to use band 1;
#     removed the input_field_rootname and polaxis options; added HLCs for bands 2-4; added
#     rotated SPC
# Version 1.2.3, 15 July 2021, John Krist: added dm_sampling_m as optional parameter
# Version 1.2.8, 11 Nov 2021, John Krist: Added option for spc-spec_band2_rotated; fixed rotated SPC
#     mask orientations

# experience has shown that with the Intel math libraries the program
# can hang if run in a multithreaded condition with more than one 
# thread allocated, so set to 1. This must be done before importing numpy

import os
os.environ['MKL_NUM_THREADS'] = '1' 
import numpy as np

import proper
import roman_phasec_proper
from roman_phasec_proper import trim, ffts, mft2


##########################################################################
def angle( n ):
    x = np.arange( n ) - int(n)//2
    return np.arctan2( x[:,np.newaxis], x ) * (180 / np.pi)

##########################################################################
def roman_phasec_compact( lambda_m, output_dim0, PASSVALUE={'dummy':0} ):

    # "output_dim" is used to specify the output dimension in pixels at the final image plane.
    # Computational grid sizes are hardcoded for each coronagraph.

    data_dir = roman_phasec_proper.data_dir
    if 'PASSVALUE' in locals():
        if 'data_dir' in PASSVALUE: data_dir = PASSVALUE['data_dir']

    cor_type = 'hlc'            # coronagraph type ('hlc', 'spc-spec_band2', 'spc-spec_band3', 'spc-wide')
    source_x_offset = 0         # source offset in lambda0_m/D radians (tilt applied at primary)
    source_y_offset = 0                 
    use_hlc_dm_patterns = 0     # use Dwight's HLC default DM wavefront patterns? 1 or 0
    use_dm1 = 0                 # use DM1? 1 or 0
    use_dm2 = 0                 # use DM2? 1 or 0
    dm_sampling_m = 0.9906e-3   # actuator spacing in meters
    dm1_m = np.zeros((48,48))
    dm1_xc_act = 23.5           # for 48x48 DM, wavefront centered at actuator intersections: (0,0) = 1st actuator center
    dm1_yc_act = 23.5              
    dm1_xtilt_deg = 0           # tilt around X axis (deg)
    dm1_ytilt_deg = 9.65        # effective DM tilt in deg including 9.65 deg actual tilt and pupil ellipticity
    dm1_ztilt_deg = 0           # rotation of DM about optical axis (deg)
    dm2_m = np.zeros((48,48))
    dm2_xc_act = 23.5           # for 48x48 DM, wavefront centered at actuator intersections: (0,0) = 1st actuator center
    dm2_yc_act = 23.5              
    dm2_xtilt_deg = 0           # tilt around X axis (deg)
    dm2_ytilt_deg = 9.65        # effective DM tilt in deg including 9.65 deg actual tilt and pupil ellipticity
    dm2_ztilt_deg = 0           # rotation of DM about optical axis (deg)
    hlc_dm1_file = ''
    hlc_dm2_file = ''
    use_fpm = 1                
    final_sampling_lam0 = 0     # final sampling in lambda0/D
    output_dim = output_dim0    # dimension of output in pixels (overrides output_dim0)

    if 'PASSVALUE' in locals():
        if 'cor_type' in PASSVALUE: cor_type = PASSVALUE['cor_type']
        if 'use_fpm' in PASSVALUE: use_fpm = PASSVALUE['use_fpm']

    is_hlc = False
    is_spc = False

    if cor_type.find('hlc') != -1:
        is_hlc = True
        if cor_type == 'hlc' or cor_type == 'hlc_band1':
            file_directory = data_dir + '/hlc_20190210b/'         # must have trailing "/"
            lambda0_m = 0.575e-6
            hlc_dm1_file = file_directory + 'hlc_dm1.fits'
            hlc_dm2_file = file_directory + 'hlc_dm2.fits'
        elif cor_type == 'hlc_band2':
            file_directory = data_dir + '/hlc_20200617c_band2/'   # must have trailing "/"
            lambda0_m = 0.660e-6
        elif cor_type == 'hlc_band3':
            file_directory = data_dir + '/hlc_20200614b_band3/'   # must have trailing "/"
            lambda0_m = 0.730e-6
        elif cor_type == 'hlc_band4':
            file_directory = data_dir + '/hlc_20200609b_band4/'   # must have trailing "/"
            lambda0_m = 0.825e-6
        else:
            print("Unsupported HLC mode")
            raise Exception(' ')
        pupil_diam_pix = 309.0
        pupil_file = file_directory + 'pupil.fits'
        # find nearest available FPM wavelength that matches specified wavelength
        if use_fpm != 0:
            lam_um = lambda_m * 1e6
            # find nearest available FPM wavelength that matches specified wavelength
            f = open( file_directory + 'fpm_files.txt' )
            fpm_nlam = int(f.readline())
            fpm_lam_um = np.zeros((fpm_nlam),dtype=float)
            for j in range(0,fpm_nlam):
                fpm_lam_um[j] = float(f.readline())
            fpm_lams = [j.strip() for j in f.readlines()]
            f.close()
            diff = np.abs(fpm_lam_um - lam_um)
            w = np.argmin( diff )
            if diff[w] > 0.0001:
                print("Error in roman_phasec_compact: requested wavelength not within 0.1 nm of nearest available FPM wavelength.")
                print("  requested (um) = " + str(lam_um) + "  closest available (um) = " + str(fpm_lam_um[w]) )
                raise Exception(' ')
            fpm_rootname = file_directory + fpm_lams[w]
            (r, header) = proper.prop_fits_read( fpm_rootname+'real.fits', header=True )
            i = proper.prop_fits_read( fpm_rootname+'imag.fits' )
            fpm_array = r + i * 1j
            fpm_mask = (r != r[0,0]).astype(int)
            fpm_sampling_lam0divD = header['FPMDX']     # FPM sampling in fpm_lam0_m/D units
            fpm_lam0_m = header['FPMLAM0M']             # FPM reference wavelength
        lyot_stop_file = file_directory + 'lyot_rotated.fits'
        n_small = 1024              # gridsize in non-critical areas
        n_big = 1024                # gridsize to FPM (propagation to/from FPM handled by MFT)
    elif cor_type == 'spc-spec' or cor_type == 'spc-spec_band2' or cor_type == 'spc-spec_band3':
        is_spc = True
        file_dir = data_dir + '/spc_20200617_spec/' # must have trailing "/"
        pupil_diam_pix = 1000.0     # Y axis pupil diameter in pixels
        pupil_file = file_dir + 'pupil_SPC-20200617_1000.fits'
        pupil_mask_file = file_dir + 'SPM_SPC-20200617_1000_rounded9.fits'
        fpm_sampling = 0.05    # sampling in lambda0/D of FPM mask 
        fpm_file = file_dir + 'fpm_0.05lamD.fits' 
        if cor_type == 'spc-spec_band2':
            fpm_lam0_m = 0.66e-6
        else:
            fpm_lam0_m = 0.73e-6
        lambda0_m = fpm_lam0_m 
        lyot_stop_file = file_dir + 'LS_SPC-20200617_1000.fits'
        n_small = 2048              # gridsize in non-critical areas
        n_big = 1400                # gridsize to FPM (propagation to/from FPM handled by MFT)
    elif cor_type == 'spc-spec_rotated' or cor_type == 'spc-spec_band2_rotated' or cor_type == 'spc-spec_band3_rotated':
        is_spc = True
        file_dir = data_dir + '/spc_20200628_specrot/'  # must have trailing "/"
        pupil_diam_pix = 1000.0     # Y axis pupil diameter in pixels
        pupil_file = file_dir + 'pupil_SPC-20200628_1000.fits'
        pupil_mask_file = file_dir + 'SPM_SPC-20200628_1000.fits'
        fpm_sampling = 0.05    # sampling in lambda0/D of FPM mask 
        fpm_file = file_dir + 'FPM_SPC-20200628_res20_flip.fits'
        if cor_type == 'spc-spec_band2_rotated':
            fpm_lam0_m = 0.66e-6
        else:
            fpm_lam0_m = 0.73e-6
        lambda0_m = fpm_lam0_m 
        lyot_stop_file = file_dir + 'LS_SPC-20200628_1000_flip.fits'
        n_small = 2048              # gridsize in non-critical areas
        n_big = 1400                # gridsize to FPM (propagation to/from FPM handled by MFT)
    elif cor_type == 'spc-wide' or cor_type == 'spc-wide_band4' or cor_type == 'spc-wide_band1':
        is_spc = True
        file_dir = data_dir + '/spc_20200610_wfov/' # must have trailing "/"
        pupil_diam_pix = 1000.0      # Y axis pupil diameter in pixels
        pupil_file = file_dir + 'pupil_SPC-20200610_1000.fits'
        pupil_mask_file = file_dir + 'SPM_SPC-20200610_1000_rounded9_gray.fits'
        fpm_sampling = 0.1     # sampling in fpm_lam0_m/D of FPM mask 
        fpm_file = file_dir + 'FPM_SPC-20200610_0.1_lamc_div_D.fits'
        if cor_type == 'spc-wide_band1':
            fpm_lam0_m = 0.575e-6
        else:
            fpm_lam0_m = 0.825e-6
        lambda0_m = fpm_lam0_m        # FPM scaled for this central wavelength
        lyot_stop_file = file_dir + 'LS_SPC-20200610_1000.fits'
        n_small = 2048              # gridsize in non-critical areas
        n_big = 1400                # gridsize to FPM (propagation to/from FPM handled by MFT)
    elif cor_type == 'spc-mswc':
        is_spc = True
        file_dir = data_dir + '/spc_20200623_mswc/' # must have trailing "/"
        pupil_diam_pix = 982.0                           # Y axis pupil diameter in pixels
        pupil_file = file_dir + 'pupil_SPC-20200623_982_rotated.fits'
        pupil_mask_file = file_dir + 'SPM_SPC-20200623_982_rounded9_gray.fits'
        fpm_sampling = 0.1    # sampling in lambda0/D of FPM mask 
        fpm_file = file_dir + 'FPM_SPC-20200623_0.1_lamc_div_D.fits'
        if cor_type == 'spc-mswc_band1':
            fpm_lam0_m = 0.575e-6
        else:
            fpm_lam0_m = 0.825e-6
        lambda0_m = fpm_lam0_m        # FPM scaled for this central wavelength
        lyot_stop_file = file_dir + 'LS_SPC-20200623_982_rotated.fits'
        n_small = 2048              # gridsize in non-critical areas
        n_big = 1400                # gridsize to FPM (propagation to/from FPM handled by MFT)
    else:
        raise Exception( 'roman_phasec_compact: Unsuported cor_type: '+cor_type )

    if 'PASSVALUE' in locals():
        if 'source_x_offset' in PASSVALUE: source_x_offset = PASSVALUE['source_x_offset']
        if 'source_y_offset' in PASSVALUE: source_y_offset = PASSVALUE['source_y_offset']
        if 'use_hlc_dm_patterns' in PASSVALUE: use_hlc_dm_patterns = PASSVALUE['use_hlc_dm_patterns']
        if 'use_dm1' in PASSVALUE: use_dm1 = PASSVALUE['use_dm1'] 
        if 'dm_sampling_m' in PASSVALUE: dm_sampling_m = PASSVALUE['dm_sampling_m']
        if 'dm1_m' in PASSVALUE: dm1_m = PASSVALUE['dm1_m']
        if 'dm1_xc_act' in PASSVALUE: dm1_xc_act = PASSVALUE['dm1_xc_act']
        if 'dm1_yc_act' in PASSVALUE: dm1_yc_act = PASSVALUE['dm1_yc_act']
        if 'dm1_xtilt_deg' in PASSVALUE: dm1_xtilt_deg = PASSVALUE['dm1_xtilt_deg']
        if 'dm1_ytilt_deg' in PASSVALUE: dm1_ytilt_deg = PASSVALUE['dm1_ytilt_deg']
        if 'dm1_ztilt_deg' in PASSVALUE: dm1_ztilt_deg = PASSVALUE['dm1_ztilt_deg']
        if 'use_dm2' in PASSVALUE: use_dm2 = PASSVALUE['use_dm2']
        if 'dm2_m' in PASSVALUE: dm2_m = PASSVALUE['dm2_m']
        if 'dm2_xc_act' in PASSVALUE: dm2_xc_act = PASSVALUE['dm2_xc_act']
        if 'dm2_yc_act' in PASSVALUE: dm2_yc_act = PASSVALUE['dm2_yc_act']
        if 'dm2_xtilt_deg' in PASSVALUE: dm2_xtilt_deg = PASSVALUE['dm2_xtilt_deg']
        if 'dm2_ytilt_deg' in PASSVALUE: dm2_ytilt_deg = PASSVALUE['dm2_ytilt_deg']
        if 'dm2_ztilt_deg' in PASSVALUE: dm2_ztilt_deg = PASSVALUE['dm2_ztilt_deg']
        if 'final_sampling_lam0' in PASSVALUE: final_sampling_lam0 = PASSVALUE['final_sampling_lam0']
        if 'output_dim' in PASSVALUE: output_dim = PASSVALUE['output_dim']

    if use_hlc_dm_patterns != 0 and cor_type != 'hlc' and cor_type != 'hlc_band1':
        raise Exception('ERROR: Can only utilize use_hlc_dm_patterns with Band 1 HLC')

    diam_at_dm1 = 0.0463
    d_dm1_dm2 = 1.0

    n = n_small
 
    wavefront = proper.prop_begin( diam_at_dm1, lambda_m, n, float(pupil_diam_pix)/n )
    pupil = proper.prop_fits_read( pupil_file )
    proper.prop_multiply( wavefront, trim(pupil,n) )
    pupil = 0
    proper.prop_define_entrance( wavefront )
    if source_x_offset != 0 or source_y_offset != 0:
        # compute tilted wavefront to offset source by xoffset,yoffset lambda0_m/D
        xtilt_lam = -source_x_offset * lambda0_m / lambda_m
        ytilt_lam = -source_y_offset * lambda0_m / lambda_m
        x = np.tile( (np.arange(n)-n//2)/(pupil_diam_pix/2.0), (n,1) )
        y = np.transpose(x)
        proper.prop_multiply( wavefront, np.exp(complex(0,1) * np.pi * (xtilt_lam * x + ytilt_lam * y)) )
        x = 0
        y = 0
    
    if is_hlc == True and use_hlc_dm_patterns == 1 and hlc_dm1_file != '':
        hlc_dm1 = proper.prop_fits_read( hlc_dm1_file )
        dm1 = dm1_m + hlc_dm1
        use_dm1 = 1
        hlc_dm2 = proper.prop_fits_read( hlc_dm2_file )
        dm2 = dm2_m + hlc_dm2
        use_dm2 = 1
    else:
        dm1 = dm1_m
        dm2 = dm2_m
    if use_dm1 != 0: proper.prop_dm( wavefront, dm1, dm1_xc_act, dm1_yc_act, dm_sampling_m, XTILT=dm1_xtilt_deg, YTILT=dm1_ytilt_deg, ZTILT=dm1_ztilt_deg )

    proper.prop_propagate( wavefront, d_dm1_dm2, 'DM2' )
    if use_dm2 != 0: proper.prop_dm( wavefront, dm2, dm2_xc_act, dm2_yc_act, dm_sampling_m, XTILT=dm2_xtilt_deg, YTILT=dm2_ytilt_deg, ZTILT=dm2_ztilt_deg )
    if is_hlc == True:
        dm2mask = proper.prop_fits_read( file_directory+'dm2mask.fits' )
        proper.prop_multiply( wavefront, trim(dm2mask, n) )
        dm2mask = 0

    proper.prop_propagate( wavefront, -d_dm1_dm2, 'back to DM1' )

    (wavefront, sampling_m) = proper.prop_end( wavefront, NOABS=True )
    n = n_big
    wavefront = trim(wavefront, n)

    # apply shaped pupil mask

    if is_spc == True:
        pupil_mask = proper.prop_fits_read( pupil_mask_file )
        wavefront *= trim(pupil_mask,n)
        pupil_mask = 0

    # propagate to FPM and apply FPM

    if use_fpm:
        if is_hlc == True:
            # use MFTs to use super-sampled FPM
            wavefront *= fpm_array[0,0]                    # apply amplitude & phase from FPM clear area
            nfpm = fpm_array.shape[0]
            fpm_sampling_lamdivD = fpm_sampling_lam0divD * fpm_lam0_m / lambda_m    # FPM sampling at current wavelength in lambda_m/D
            wavefront_fpm = mft2(wavefront, fpm_sampling_lamdivD, pupil_diam_pix, nfpm, +1)   # MFT to highly-sampled focal plane
            wavefront_fpm = wavefront_fpm * fpm_mask * (fpm_array - 1)      # subtract field inside FPM region, add in FPM-multiplied region
            wavefront_fpm = mft2(wavefront_fpm, fpm_sampling_lamdivD, pupil_diam_pix, n, -1)        # MFT back to pupil (Lyot stop)
            wavefront += wavefront_fpm
            wavefront_fpm = 0
        elif is_spc == True:
            fpm = proper.prop_fits_read( fpm_file )
            nfpm = fpm.shape[1]
            fpm_sampling_lam = fpm_sampling * fpm_lam0_m / lambda_m
            wavefront = mft2(wavefront, fpm_sampling_lam, pupil_diam_pix, nfpm, +1)   # MFT to highly-sampled focal plane
            wavefront *= fpm
            fpm = 0
            wavefront = mft2(wavefront, fpm_sampling_lam, pupil_diam_pix, int(pupil_diam_pix), -1)  # MFT to Lyot stop 

    n = n_small
    wavefront = trim(wavefront,n)
    lyot = proper.prop_fits_read( lyot_stop_file )
    wavefront *= trim(lyot,n)
    lyot = 0

    wavefront *= n
    wavefront = ffts(wavefront,-1)    # to focus

    # rotate to convention used by full prescription

    wavefront[:,:] = np.rot90( wavefront, 2 )
    wavefront[:,:] = np.roll( wavefront, 1, axis=0 ) 
    wavefront[:,:] = np.roll( wavefront, 1, axis=1 )
 
    if final_sampling_lam0 != 0:
        mag = (float(pupil_diam_pix)/n) / final_sampling_lam0 * (lambda_m/lambda0_m)
        wavefront = proper.prop_magnify( wavefront, mag, output_dim, AMP_CONSERVE=True )
    else:
        wavefront = trim(wavefront, output_dim)

    sampling_m = 0.0
    return wavefront, sampling_m

