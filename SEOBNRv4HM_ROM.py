import copy
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import h5py
import numpy as np
import TPI
import TPI
import Conditioning_SEOBNRv4HM_FD as conditioning
import lal
import lalsimulation as ls
import scipy
import time

# Definition of the functions

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Logarithmic sampling
def logspace(start, stop, nb):
    ratio = (stop/start)**(1./(nb-1))
    return start * np.power(ratio, np.arange(nb))

def import_ROM_data_from_hdf5(hdf5_file_path, modes = [(2,2),(2,1),(3,3),(4,4),(5,5)]):
    
    
    # Open hdf5 file
    file_ROM_hdf5 = h5py.File(hdf5_file_path,'r')
    
    # Read the patches available for the ROM
    patches = file_ROM_hdf5.keys()
    
    # Define output dictionaries
    grid_k_p, grid_hlm_p, B_phi_k_p, B_hlm_coorb_Re_p, B_hlm_coorb_Im_p, Iphi_k_p, Ihlm_coorb_Re_p, Ihlm_coorb_Im_p = {}, {}, {}, {}, {}, {}, {}, {} 
    
    for patch in patches:
    
        # Import the grid
        
        grid_q = file_ROM_hdf5[patch]['qvec'][()]
        grid_chi1 = file_ROM_hdf5[patch]['chi1vec'][()]
        grid_chi2 = file_ROM_hdf5[patch]['chi2vec'][()]
        X = [grid_q,grid_chi1,grid_chi2] 
        
        # Import the ROM data
        
        # Import grid, basis and projection coefficients for the carrier phase
        grid_k = {}
        B_phi_k = {}
        Iphi_k_coeffs = {}
        
        grid_k = file_ROM_hdf5[patch]['phase_carrier']['MF_grid'][()]
        B_phi_k = file_ROM_hdf5[patch]['phase_carrier']['basis'][()]
        Iphi_k_coeffs = file_ROM_hdf5[patch]['phase_carrier']['coeff'][()]
        
        # Import grid, basis and projection coefficients of the co-orbital frame modes
        grid_hlm = {}
        B_hlm_coorb_Re = {}
        B_hlm_coorb_Im = {}
        Ihlm_coorb_Re_coeffs = {}
        Ihlm_coorb_Im_coeffs = {}
        
        for lm in modes:
            grid_hlm[lm] = file_ROM_hdf5[patch]['CF_modes'][str(lm[0])+str(lm[1])]['MF_grid'][()]
            B_hlm_coorb_Re[lm] = file_ROM_hdf5[patch]['CF_modes'][str(lm[0])+str(lm[1])]['basis_re'][()]
            B_hlm_coorb_Im[lm] = file_ROM_hdf5[patch]['CF_modes'][str(lm[0])+str(lm[1])]['basis_im'][()]
            Ihlm_coorb_Re_coeffs[lm] = file_ROM_hdf5[patch]['CF_modes'][str(lm[0])+str(lm[1])]['coeff_re'][()]
            Ihlm_coorb_Im_coeffs[lm] = file_ROM_hdf5[patch]['CF_modes'][str(lm[0])+str(lm[1])]['coeff_im'][()]
        
        

        # Unpack ROM data
        N_freqs = len(Iphi_k_coeffs)

        Ihlm_coorb_Re = {}
        Ihlm_coorb_Im = {}
        Iphi_k = []
        for lm in modes:
            Ihlm_coorb_Re[lm] = []
            Ihlm_coorb_Im[lm] = []
        for i in range(N_freqs):
            Iphi_k.append(TPI.TP_Interpolant_ND(X,coeffs=Iphi_k_coeffs[i]))
            for lm in modes:
                Ihlm_coorb_Re[lm].append(TPI.TP_Interpolant_ND(X,coeffs=Ihlm_coorb_Re_coeffs[lm][i])) 
                Ihlm_coorb_Im[lm].append(TPI.TP_Interpolant_ND(X,coeffs=Ihlm_coorb_Im_coeffs[lm][i]))       

        # Save all the needed data
        
        grid_k_p[patch] = grid_k
        grid_hlm_p[patch] = grid_hlm
        B_phi_k_p[patch] = B_phi_k
        B_hlm_coorb_Re_p[patch] = B_hlm_coorb_Re
        B_hlm_coorb_Im_p[patch] = B_hlm_coorb_Im
        Iphi_k_p[patch] = Iphi_k
        Ihlm_coorb_Re_p[patch] = Ihlm_coorb_Re
        Ihlm_coorb_Im_p[patch] = Ihlm_coorb_Im
    
    # Close hdf5 file
    file_ROM_hdf5.close()
    
    return grid_k_p, grid_hlm_p, B_phi_k_p, B_hlm_coorb_Re_p, B_hlm_coorb_Im_p, Iphi_k_p, Ihlm_coorb_Re_p, Ihlm_coorb_Im_p

def rescale_factor_amplitude(q,chi1,chi2,l,m,freq):
    eta = q/(1+q)**2
    delta = np.sqrt(1.-4.*eta)
    chiS = (chi1+chi2)/2.
    chiA = (chi1-chi2)/2.
    v = (2*np.pi*freq/2.)**(1./3.)
    H22 = -1 + (323./224. -451.*eta/168.)*v**2
    if l == 3:
        if m == 3:
            v33 = (2*np.pi*freq/3.)**(1./3.)
            return (m/2.)**(7./6.)*(-3./4. * np.sqrt(5./7.) * (delta*v33 + delta*(-1945./672.+27.*eta/8.)*v33**3 
            +(chiA*(161./24. - 85.*eta/3.) +delta*chiS*(161./24. -17.*eta/3.))*v33**4 ))/H22
    if l == 4:
        if m == 4:
            v44 = (2*np.pi*freq/4.)**(1./3.)
            return (m/2.)**(7./6.)*(np.sqrt(10./7.)*(-4./9.)*(1.-3.*eta)*v44**2 )/H22
    if l == 2:
        if m ==1:
            v21 = (2*np.pi*freq)**(1./3.)
            return (m/2.)**(7./6.)*(-np.sqrt(2.)/3.)* (delta * v21 + (-3./2.)*(chiA + delta*chiS)*v21**2)/H22
    if l == 5:
        if m == 5:
            v55 = (2*np.pi*freq/5.)**(1./3.)
            return (m/2.)**(7./6.)*(np.sqrt(5./33.))*(-125./96.)*delta*(1.-2*eta)*v55**3/H22

def Get_omegaQNM_SEOBNRv4(q, chi1z, chi2z, lm):
    M = 100. # will be ignored
    Ms = M * lal.MTSUN_SI
    m1 = M * q/(1+q)
    m2 = M * 1/(1+q)
    complexQNM = lal.CreateCOMPLEX16Vector(1)
    ls.SimIMREOBGenerateQNMFreqV2(complexQNM, m1, m2, np.array([0., 0., chi1z]), np.array([0., 0., chi2z]), lm[0], lm[1], 1, ls.SEOBNRv4)
    return Ms * np.real(complexQNM.data[0])

def blend(f, f1, f2):
    return np.piecewise(f,[f<=f1,np.logical_and(f1 < f, f < f2),f>=f2],[lambda f: 0, lambda f: scipy.special.expit(- (f2 - f1)/(f-f1) - (f2 - f1)/(f-f2)), lambda f: 1])     

def blend_functions(freq_array_1, freq_array_2, fun_1_in,fun_2_in, f_hyb_geo_start, f_hyb_geo_end):
    # This function takes a LF and an HF functions and blends them together
    # Here 1 shold be the LF function and 2 the HF function
    
    if freq_array_1[-1]<f_hyb_geo_end:
        raise ValueError('The last frequency of the LF array = {} should be > than f_hyb_geo_end = {}'.format(freq_array_1[-1],f_hyb_geo_end))
    if freq_array_2[0]>f_hyb_geo_start:
        raise ValueError('The first frequency of the HF array = {} should be < than f_hyb_geo_start = {}'.format(freq_array_2[0],f_hyb_geo_start)) 
    
     
    fun_1 = copy.deepcopy(fun_1_in)
    fun_2 = copy.deepcopy(fun_2_in)

    #freq_array = np.concatenate((freq_array_1[freq_array_1 < f_hyb_geo_start],np.linspace(f_hyb_geo_start,f_hyb_geo_end,num=10),freq_array_2[freq_array_2 > f_hyb_geo_end]))
    freq_array = np.concatenate((freq_array_1[freq_array_1 < f_hyb_geo_start],freq_array_2[freq_array_2 >= f_hyb_geo_start]))

    fun_1 = spline(freq_array_1,fun_1,ext=1)(freq_array)

    fun_2 = spline(freq_array_2,fun_2,ext=1)(freq_array)

    bld_2 = blend(freq_array,f_hyb_geo_start,f_hyb_geo_end)
    bld_1 = 1. - bld_2
    bld_1[freq_array>f_hyb_geo_end] = 0.
    bld_2[freq_array<f_hyb_geo_start] = 0.

    fun = fun_1*bld_1 + fun_2*bld_2

    return freq_array, fun  

def get_sparse_frequencies_rescaled(freq_in,q,chi1,chi2,lm):
    freq = copy.deepcopy(freq_in)
    omegaQNM = Get_omegaQNM_SEOBNRv4(q, chi1, chi2, lm)
    inv_scaling = 1. / (2*np.pi/omegaQNM)
    return freq*inv_scaling

def select_HF_patch(q,chi1):
    if (q> 3.) and (chi1<=0.8):
        patch = 'hqls'
    if (q> 3.) and (chi1>0.8):
        patch = 'hqhs'
    if (q<= 3.) and (chi1<0.8):
        patch = 'lqls'      
    if (q<= 3.) and (chi1>0.8):
        patch = 'lqhs'
    return patch    

# This function returns the phase of the carrier in the sparse grid
def carrier_phase_sparse_grid(q,chi1,chi2,freq_range):
    # Select the ROM patch
    if freq_range == 'HF':
        patch = select_HF_patch(q,chi1)
    elif freq_range == 'LF':
        patch = 'lowf'
    else:
        raise ValueError('Patch not available')
    # Compute the phase           
    cphi_k = [I([q, chi1, chi2]) for I in Iphi_k_patches[patch]]
    phi_k = np.dot(cphi_k, B_phi_k_patches[patch])

    # Get sparse frequency array
    sparse_freq = grid_k_patches[patch]
    if freq_range == 'HF':
        # Get the unscaled frequency
        sparse_freq = get_sparse_frequencies_rescaled(sparse_freq,q,chi1,chi2,(2,2))   

    return sparse_freq, phi_k
# This function returns the co-orbital frame modes in the sparse grid
def co_orbital_modes_sparse_grid(q,chi1,chi2, freq_range, modes = [(2,2),(2,1),(3,3),(4,4),(5,5)]):
    # Select the ROM patch
    if freq_range == 'HF':
        patch = select_HF_patch(q,chi1)
    elif freq_range == 'LF':
        patch = 'lowf'
    else:
        raise ValueError('Patch not available')

    # Get co-orbital frame modes in sparse array    
    chlm_coorb_Re = {}
    chlm_coorb_Im = {}
    hlm_coorb_Re = {}
    hlm_coorb_Im = {}
    sparse_freq_lm = {}
    for lm in modes:
        chlm_coorb_Re[lm] = [I([q, chi1, chi2]) for I in Ihlm_coorb_Re_patches[patch][lm]]
        chlm_coorb_Im[lm] = [I([q, chi1, chi2]) for I in Ihlm_coorb_Im_patches[patch][lm]]   
        hlm_coorb_Re[lm] = np.dot(chlm_coorb_Re[lm], B_hlm_coorb_Re_patches[patch][lm])
        hlm_coorb_Im[lm] = np.dot(chlm_coorb_Im[lm], B_hlm_coorb_Im_patches[patch][lm])
        # Get sparse frequency array
        sparse_freq_lm[lm] = grid_hlm_patches[patch][lm]
        if freq_range == 'HF':
        # Get the unscaled frequency
            sparse_freq_lm[lm] = get_sparse_frequencies_rescaled(sparse_freq_lm[lm],q,chi1,chi2,lm)
    return sparse_freq_lm, hlm_coorb_Re, hlm_coorb_Im

def phase_carrier_hybrid(q, chi1, chi2):
    highf_array, phase_carrier_HF = carrier_phase_sparse_grid(q, chi1, chi2,'HF')
    lowf_array, phase_carrier_LF = carrier_phase_sparse_grid(q, chi1, chi2,'LF')    
    f_array, phase_carrier = blend_functions(lowf_array, highf_array, phase_carrier_LF,phase_carrier_HF, f_hyb_ini, f_hyb_end)
    return f_array, phase_carrier

def co_orbital_modes_hybrid(q,chi1,chi2,modes = [(2,2),(2,1),(3,3),(4,4),(5,5)]):
    f_array_modes = {}
    modes_Re, modes_Im = {}, {}
    lowf_array_modes, modes_Re_LF, modes_Im_LF =  co_orbital_modes_sparse_grid(q,chi1,chi2, 'LF',modes)
    highf_array_modes, modes_Re_HF, modes_Im_HF =  co_orbital_modes_sparse_grid(q,chi1,chi2, 'HF',modes)
    for lm in modes:
        f_array_modes[lm], modes_Re[lm] =  blend_functions(lowf_array_modes[lm], highf_array_modes[lm], modes_Re_LF[lm],modes_Re_HF[lm], lm[1]*f_hyb_ini, lm[1]*f_hyb_end)
        f_array_modes[lm], modes_Im[lm] =  blend_functions(lowf_array_modes[lm], highf_array_modes[lm], modes_Im_LF[lm],modes_Im_HF[lm], lm[1]*f_hyb_ini, lm[1]*f_hyb_end)
    return f_array_modes, modes_Re, modes_Im

def get_modes(q, chi1, chi2,modes = [(2,2), (2,1), (3,3), (4,4), (5,5)],debug=False):

    if debug==True:
        start = time.time()

    # Get the hybridize phase carrier

    freq_carrier, phase_carrier = phase_carrier_hybrid(q, chi1, chi2)

    # Get the modes in the co-orbital frame

    f_array_modes, modes_Re, modes_Im = co_orbital_modes_hybrid(q,chi1,chi2)

    # Get the FD orb phase change to be undone

    # NOTE: allow extrapolation to avoid machine-precision out-of-bounds errors
    Psi_k_int = spline(freq_carrier, -phase_carrier, ext=0)

    f_max_k = freq_carrier[-1]
    Psi_k_end = Psi_k_int(f_max_k)
    dfPsi_k_end = Psi_k_int(f_max_k, 1)
    def func_Psi_k_ext(f):
        f_k = f[f<=f_max_k]
        f_ext = f[f>f_max_k]
        Psi_k = np.zeros(len(f), dtype=float)
        Psi_k[f<=f_max_k] = Psi_k_int(f_k)
        Psi_k[f>f_max_k] = Psi_k_end + (f_ext-f_max_k) * dfPsi_k_end
        return Psi_k

    Deltaphi_lm = {}

    Deltaphi_lm[(2,2)] = 0.
    Deltaphi_lm[(2,1)] = np.pi/2
    Deltaphi_lm[(3,3)] = -np.pi/2
    Deltaphi_lm[(4,4)] = np.pi
    Deltaphi_lm[(5,5)] = np.pi/2

    Psi_lm_approx = {}
    for lm in modes:
        m = lm[1]
        Psi_lm_approx[lm] = m*func_Psi_k_ext(1./m*f_array_modes[lm]) + Deltaphi_lm[lm] + (1 - m)*np.pi/4

    #---

    # Get the modes

    phaselm_undo = {}
    amplm_undo = {}
    for lm in modes:
        phaselm = np.unwrap(np.angle(modes_Re[lm] +1j*modes_Im[lm])) -Psi_lm_approx[lm]
        amplm = np.abs(modes_Re[lm] +1j*modes_Im[lm])

        # Limit the ROM to the available high frequency content
        f_end_geo = ratiosQNM_highfcut[lm]*Get_omegaQNM_SEOBNRv4(q,chi1,chi2,lm)/(2.*np.pi)
        amplm[f_array_modes[lm]>f_end_geo] = 0.
        phaselm[f_array_modes[lm]>f_end_geo] = 0.
        phaselm_undo[lm] = phaselm
        amplm_undo[lm] = amplm


    if debug==True:
        end = time.time()
        print('Function get_modes executed in {}s'.format(end-start))
    return f_array_modes, amplm_undo, phaselm_undo

def get_modes_sampled_freqs_old(freq, q, chi1, chi2,modes = [(2,2), (2,1), (3,3), (4,4), (5,5)],debug=False):
    
    if debug==True:
        start = time.time()

    f_array_modes, amplm_undo, phaselm_undo = get_modes(q, chi1, chi2,modes = modes,debug=debug)
    hlm = {}
    for lm in modes:
        amplm_undo[lm] = spline(f_array_modes[lm],amplm_undo[lm],ext=1)(freq[lm])
        phaselm_undo[lm] = spline(f_array_modes[lm],phaselm_undo[lm],ext=1)(freq[lm])
        hlm[lm] = amplm_undo[lm]*np.exp(1j*phaselm_undo[lm])        
    if debug==True:
        end = time.time()
        print('Function get_modes_sampled_freqs executed in {}s'.format(end-start))
    return hlm 

def get_modes_sampled_freqs(freq, f_array_modes, amplm, phaselm, modes = [(2,2), (2,1), (3,3), (4,4), (5,5)],debug=False):
    
    if debug==True:
        start = time.time()

    hlm = {}
    for lm in modes:
        amplm[lm] = spline(f_array_modes[lm],amplm[lm],ext=1)(freq[lm])
        phaselm[lm] = spline(f_array_modes[lm],phaselm[lm],ext=1)(freq[lm])
        hlm[lm] = amplm[lm]*np.exp(1j*phaselm[lm])        
    if debug==True:
        end = time.time()
        print('Function get_modes_sampled_freqs executed in {}s'.format(end-start))
    return hlm  


def get_TF2_22mode(q, chi1, chi2, f_ini_geo, f_end_geo, delta_f_geo):
    
    #Set intrinsic and extrinsic parameters
    M_TF2 = 1.
    deltaF_Hz = delta_f_geo/(lal.MTSUN_SI*M_TF2)
    fini_Hz = f_ini_geo/(lal.MTSUN_SI*M_TF2)
    m1 = M_TF2*q*lal.MSUN_SI/(1+q)
    m2 = M_TF2*lal.MSUN_SI/(1+q)
    approx = ls.SimInspiralGetApproximantFromString('TaylorF2')
    distance = 1e9*lal.PC_SI
    Mr = (m1+m2) * lal.MRSUN_SI/lal.MSUN_SI
    phiRef = 0.
    fRef = fini_Hz
    fmax_Hz = f_end_geo/(lal.MTSUN_SI*M_TF2)
    
    #Generate the waveform
    hp, hc = ls.SimInspiralChooseFDWaveform(m1,m2,0,0,chi1,0,0,chi2,distance,0.,phiRef,0.,0.,0.,deltaF_Hz,fini_Hz,fmax_Hz,fRef,None,approx)
    f = np.arange(hp.data.data.size) * delta_f_geo
    
    # Extract the 22 mode from TaylorF2 and SEOBNRv4_ROM
    strain = hp.data.data +1j*hc.data.data
    h22_geo = strain*(distance/Mr)*(1./(lal.MTSUN_SI*M_TF2*lal.SpinWeightedSphericalHarmonic(0.,0.,-2,2,2)))
    amp22_geo, phase22_geo = np.abs(h22_geo), -np.unwrap(np.angle(h22_geo))
    return f, amp22_geo, phase22_geo

def align_wfs_window(f_array_1, f_array_2, phase_1_in, phase_2_in, f_align_geo_start, f_align_geo_end):
    phase_1 = copy.deepcopy(phase_1_in)
    phase_2 = copy.deepcopy(phase_2_in)
    
    # Interpolate TF2 phase on EOB common grid
    phase_1_EOB_grid = scipy.interpolate.InterpolatedUnivariateSpline(f_array_1, phase_1,ext=1)(f_array_2)
    phase_diff = phase_1_EOB_grid-phase_2
    
    # Set initial and final index for alignment
    i_start_fit = find_nearest_idx(f_array_2, f_align_geo_start)
    i_end_fit = find_nearest_idx(f_array_2, f_align_geo_end)
    
    lin_fit = scipy.interpolate.InterpolatedUnivariateSpline(f_array_2[i_start_fit:i_end_fit], phase_diff[i_start_fit:i_end_fit])(f_array_2[i_start_fit:i_end_fit])
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(f_array_2[i_start_fit:i_end_fit],lin_fit)
    
    phase_2_aligned = phase_1 - (slope*f_array_1 + intercept)
    
    return phase_2_aligned

def TF2Hybrid_22mode(q, chi1, chi2, f_array_2_in, ampwf_2_in, phasewf_2_in, f_ini_geo, f_hyb_geo_start, f_hyb_geo_end, deltaF_geo):
    f_array_2 = copy.deepcopy(f_array_2_in)
    ampwf_2 = copy.deepcopy(ampwf_2_in)
    phasewf_2 = copy.deepcopy(phasewf_2_in)
    
    # Get TF2 22 mode
    f_array_1_unresampled, ampwf_1, phasewf_1 = get_TF2_22mode(q, chi1, chi2, f_ini_geo, 3*f_hyb_geo_end, deltaF_geo)
    f_array_1_unresampled, ampwf_1, phasewf_1  = f_array_1_unresampled[f_array_1_unresampled>=f_ini_geo], ampwf_1[f_array_1_unresampled>=f_ini_geo], phasewf_1[f_array_1_unresampled>=f_ini_geo]

    # Resemple TF2 22 mode
    f_array_1 = np.linspace(f_array_1_unresampled[0],f_array_1_unresampled[-1],num=1000)
    ampwf_1 = spline(f_array_1_unresampled,ampwf_1,ext=1)(f_array_1)
    phasewf_1 = spline(f_array_1_unresampled,phasewf_1,ext=1)(f_array_1)

    phasewf_1 = align_wfs_window(f_array_1, f_array_2, phasewf_1, phasewf_2, f_hyb_geo_start, f_hyb_geo_end)
    
    
    f_hyb, amp_hyb = blend_functions(f_array_1, f_array_2, ampwf_1,ampwf_2, f_hyb_geo_start, f_hyb_geo_end)
    _, phase_hyb =  blend_functions(f_array_1, f_array_2, phasewf_1,phasewf_2, f_hyb_geo_start, f_hyb_geo_end)

    return f_array_1, ampwf_1, phasewf_1, f_hyb, amp_hyb, phase_hyb

def pi_shift_odd_modes(f_array_1_in,f_array_2_in,phase_1_in,phase_2_in,f_start_hyb,f_end_hyb,debug=False):
    f_array_1 = copy.deepcopy(f_array_1_in)
    f_array_2 = copy.deepcopy(f_array_2_in)
    phase_1 = copy.deepcopy(phase_1_in)
    phase_2 = copy.deepcopy(phase_2_in)
    freqs = np.linspace(f_start_hyb,f_end_hyb,num=100)
    
    phase_1_int = spline(f_array_1,phase_1,ext=1)(freqs)
    phase_2_int = spline(f_array_2,phase_2,ext=1)(freqs)

    delta_phase = phase_1_int-phase_2_int
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(freqs,delta_phase)
    if debug==True:
        print intercept
    
    phase_1 = phase_1 - intercept              
    
    return phase_1    

def TF2Hybrid_HMmode(q,chi1,chi2, f_array_1_in, f_array_2_in, phase_1_in, phase_2_in, amp_1_in, amp_2_in, f_ini_geo, f_hyb_geo_start, f_hyb_geo_end, deltaF_geo,modes=[(2, 1), (3, 3), (4, 4), (5, 5)],debug=False):
    c_lm = {(2, 2): 0, (2, 1): np.pi/2., (3, 3):-np.pi/2., (4, 4):np.pi, (5, 5): np.pi/2}
    
    f_array_1 = copy.deepcopy(f_array_1_in)
    f_array_2 = copy.deepcopy(f_array_2_in)
    phase_1 = copy.deepcopy(phase_1_in)
    phase_2 = copy.deepcopy(phase_2_in)
    amp_1 = copy.deepcopy(amp_1_in)
    amp_2 = copy.deepcopy(amp_2_in)

    f_hyb_lm = {}
    for lm in modes:
        f_hyb_lm[lm] = np.arange(f_ini_geo*lm[1]/2., 3*f_hyb_geo_end*lm[1]/2.,deltaF_geo)
    
    amplm = {}
    philm = {}
    amplm_hyb = {}
    phaselm_hyb = {}

    for lm in modes:
        if debug==True:
            print lm
        philm[lm] = (lm[1]/2.) * (phase_1+np.pi/4.) - np.pi/4. +c_lm[lm]
        philm[lm] = scipy.interpolate.InterpolatedUnivariateSpline(f_array_1*lm[1]/2.,philm[lm],ext=1)(f_hyb_lm[lm])
        philm[lm] = pi_shift_odd_modes(f_hyb_lm[lm],f_array_2[lm],philm[lm],phase_2[lm],f_hyb_geo_start*lm[1]/2.,f_hyb_geo_end*lm[1]/2.,debug=debug)
        amplm[lm] = amp_1*np.abs(rescale_factor_amplitude(q,chi1,chi2,lm[0],lm[1],f_array_1))
        amplm[lm] = scipy.interpolate.InterpolatedUnivariateSpline(f_array_1,amplm[lm],ext=1)(f_hyb_lm[lm])
        _, phaselm_hyb[lm] =  blend_functions(f_hyb_lm[lm], f_array_2[lm], philm[lm],phase_2[lm], f_hyb_geo_start*lm[1]/2., f_hyb_geo_end*lm[1]/2.)
        f_hyb_lm[lm], amplm_hyb[lm] =  blend_functions(f_hyb_lm[lm], f_array_2[lm], amplm[lm],amp_2[lm], f_hyb_geo_start*lm[1]/2., f_hyb_geo_end*lm[1]/2.)
        
    return f_hyb_lm, phaselm_hyb, amplm_hyb

def get_modes_hyb(q, chi1, chi2, freqs_log, amplm, phaselm, f_ini_geo, deltaF_geo, modes = [(2,2), (2,1), (3,3), (4,4), (5,5)],debug=False):
    if debug==True:
        start=time.time()

    # Generate TF2 EOB hybrid 22 mode    
    f_array_TF2, ampwf_TF2, phasewf_TF2, f_hyb, amp_hyb, phase_hyb =  TF2Hybrid_22mode(q, chi1, chi2, freqs_log[(2,2)], amplm[(2,2)], phaselm[(2,2)], f_ini_geo, f_hyb_geo_start, f_hyb_geo_end, deltaF_geo)
    
    # We need to define them before in the case we are not asking for the hybrid with HM as well
    flm_hyb, phaselm_hyb, amplm_hyb = {}, {}, {}
    # Generate HM hybrids
    if modes != [(2,2)]:
        flm_hyb, phaselm_hyb, amplm_hyb  = TF2Hybrid_HMmode(q,chi1,chi2,f_array_TF2,freqs_log,phasewf_TF2,phaselm,ampwf_TF2,amplm,f_ini_geo,f_hyb_geo_start,f_hyb_geo_end,deltaF_geo,debug=debug)

    # Put the 22 mode in the same structure of the other HM
    flm_hyb[(2,2)], phaselm_hyb[(2,2)], amplm_hyb[(2,2)] = f_hyb, phase_hyb, amp_hyb



    if debug==True:
        end=time.time() 
        print('Function get_modes_hyb executed in {}s'.format(end-start))

    return flm_hyb, phaselm_hyb, amplm_hyb
   

def SEOBNRv4HM_ROM_modes(q, chi1, chi2, freqs, modes = [(2,2), (2,1), (3,3), (4,4), (5,5)],debug=False):
    
    # Output amp and phase
    amp_out = {}
    phase_out = {}
    freq_out = {}

    # Check if and for which modes is needed to construct the hybrid with TF2
    modes_hyb_fini = {} 
    for lm in modes:
        flm_i = freqs[lm][0]
        if flm_i < f_low[lm]:
            modes_hyb_fini[lm] = flm_i
        
    # Generate the LF +  HF ROM
    freqs_log, amplm, phaselm  = get_modes(q,chi1,chi2,modes=modes,debug=debug)

    # Generate TF2 hybrid if needed
    if modes_hyb_fini:
        # Check the f_ini for TF2 such that we get the lm hybrid at the right initial frequency
        f_ini_TF2 = np.min([(2./lm[1])*modes_hyb_fini[lm] for lm in modes_hyb_fini.keys()])
        # Get the delta_F, assuming is the same for all the modes
        deltaF_geo = freqs[(2,2)][1]-freqs[(2,2)][0]
        freq_out, phase_out, amp_out = get_modes_hyb(q, chi1, chi2, freqs_log, amplm, phaselm, f_ini_TF2, deltaF_geo, modes = modes_hyb_fini.keys(),debug=debug)



    # Insert the modes that don't need to be hybridized into the output dictionary
    for lm in modes:
        if lm not in modes_hyb_fini.keys():
            freq_out[lm] = freqs_log[lm]
            amp_out[lm] = amplm[lm]
            phase_out[lm] = phaselm[lm]

    # Resemple the output modes        
    hlm = get_modes_sampled_freqs(freqs, freq_out, amp_out, phase_out, modes = modes,debug=debug)
    return hlm

# Import ROM data

grid_k_patches, grid_hlm_patches, B_phi_k_patches, B_hlm_coorb_Re_patches, B_hlm_coorb_Im_patches, Iphi_k_patches, Ihlm_coorb_Re_patches, Ihlm_coorb_Im_patches = import_ROM_data_from_hdf5('/home/rcotesta/SEOBNRv4HM_ROM/SEOBNRv4HM_ROM/src/scripts/SEOBNRv4HMROM.hdf5')

# Definition of hybridization frequencies for the LF and HF ROM
f_hyb_ini = 0.0045
f_hyb_end = 0.005

# Definition of hybridization frequencies between TF2 and LF ROM
f_hyb_geo_start = 0.0005
f_hyb_geo_end = f_hyb_geo_start*1.15

# High frequency limits of the modes

ratiosQNM_highfcut = {}
ratiosQNM_highfcut[(2,2)] = 1.7
ratiosQNM_highfcut[(2,1)] = 1.7
ratiosQNM_highfcut[(3,3)] = 1.55
ratiosQNM_highfcut[(4,4)] = 1.35
ratiosQNM_highfcut[(5,5)] = 1.25

# Definition of the LF limits for the pure ROM
f_low = {}
f_low[(2,2)] = 0.0004925491025543576
f_low[(3,3)] = 0.0004925491025543576*(3./2.)
f_low[(4,4)] = 0.0004925491025543576*(4./2.)
f_low[(2,1)] = 0.0004925491025543576*(1./2.)
f_low[(5,5)] = 0.0004925491025543576*(5./2.)