import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import k,h,c
from scipy.optimize import curve_fit
from numba import njit,jit
import emcee

@njit
def Planck(lamb,T):
    '''
    Black-body radiation
    '''
    flux = 2*h*c**2/lamb**5 * 1/ (np.exp(h*c/(lamb*k*T))-1)
    return flux

def get_mag_filtered(filter_data,band,T2,b):
    '''
    Returns calibrated magnitude based on the difference between
    two black-body curves at two different temperatures (Vega and target).
    The difference of the distance between two objects is absorbed into the free parameter b.
    '''
    # select filter-specific weights and wavelengths
    band_response = filter_data[band]

    x_th = band_response[0] * 1e-10 # angstrom to meter
    weights = band_response[1]
    
    # get continuous flux curve
    flux_vega = Planck(x_th,9602)
    flux_test = Planck(x_th,T2)
    
    # add weight and take mean
    flux_vega_binned = (flux_vega*weights).sum()
    flux_test_binned = (flux_test*weights).sum()
        
    # calculate mag
    mag = -2.5*np.log10(flux_test_binned/flux_vega_binned) + b
    return mag

def calc_loglik(T_test,filter_data,bands,mag_obs,mag_err):
    '''
    Calculate log-likelihood
    '''
    mag_fit = np.asarray(list(map(
        lambda band: get_mag_filtered(filter_data,band,T_test,0),
        bands)))
    b = np.average(mag_obs-mag_fit, weights=1/mag_err**2)
    chi2 = (mag_obs-mag_fit-b)**2/mag_err**2
    loglik = -0.5* (chi2 + 2*np.pi*mag_err**2).sum()
    return loglik

def log_Gaussian(x,mu,sigma,offset):
    return -0.5*(x-mu)**2/sigma**2 + offset

def get_Temp(filter_data,bands,m_obs,m_err,T_min=3000,T_max=15000,dT=10,cut_range=1000,R_peak=500,Nsigma_range=2,multiprocessing=True):
    '''
    Calculate temperature.
    '''
    global helper
    def helper(Temp):
        return calc_loglik(Temp,filter_data,bands,m_obs,m_err)

    ## large search
    temps = np.linspace(T_min,T_max,int((T_max-T_min)/dT))
    if multiprocessing:
        with Pool() as pool:
            loglik = pool.map(helper,temps)
            pool.close()
            pool.join()
    else:
        loglik = list(map(helper,temps))
    loglik = np.asarray(loglik)

    ## estimate uncertainty
    mpv = temps[loglik==loglik.max()][0]
    cut = (temps>mpv-cut_range) & (temps<mpv+cut_range)
    popt,_ = curve_fit(lambda x,sigma,offset:log_Gaussian(x,mpv,sigma,offset),temps[cut],loglik[cut])
    T_sigma = popt[0]
    if T_sigma<dT:
        print('warning: results may not be accurate. Try again with smaller dT (default 10)')

    ## fine search
    temps = np.linspace(mpv-Nsigma_range*T_sigma,
                        mpv+Nsigma_range*T_sigma,
                        R_peak)
    if multiprocessing:
        with Pool() as pool:
            loglik = pool.map(helper,temps)
            pool.close()
            pool.join()
    else:
        loglik = list(map(helper,temps))
    loglik = np.asarray(loglik)
    popt,_ = curve_fit(log_Gaussian,temps,loglik,p0=[mpv,T_sigma,loglik.max()])
    T_mpv = popt[0]
    T_sigma = popt[1]

    return T_mpv,T_sigma