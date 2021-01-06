import numpy as np
from scipy.optimize import curve_fit
import numba
from multiprocessing import Pool


@numba.njit
def fourier(x,period,Nterms,params,debug=False):
    '''
    A Fourier function (model) that calculates y-value 
    at each x-value for given period and parametrs.
    ** IMPORTANT NOTE: ```params``` has to be a numpy array (not python list)
    '''
    if debug:
        print('*Fourier: starting Fourier')
        print('*Fourier: params = ',params)
    y = np.ones_like(x) * params[0]
    C_list = params[1:Nterms+1]
    phi_list = params[Nterms+1:]
    if debug:
        print('*Fourier: y_initial = ',y)
        print('*Fourier: C_list = ',C_list)
        print('*Fourier: phi_list = ',phi_list)
    for i in range(Nterms):
        y = y + C_list[i] * np.cos((i+1)*2*np.pi*x/period + phi_list[i])
    if debug:
        print('*Fourier: y after calculation = ',y)
    return y

def get_bestfit_Fourier(x,y,yerr,period,Nterms,return_yfit=True,return_params=False,
                        debug=False):
    '''
    ### Fourier Model ###
    returns best-fit y-values at given x
    if return_yfit==True, it returns best-fit y-values at given x
    if return_params==True, it returns best-fit parameters (model-dependent)
    NOTE: Fourier parameters are not bound to keep the code fast.
    For stellar parameter calculation purpose, use tools in StellarModels class.
    '''
    if debug:
        print('*get_bestfit_Fourier: starting process get_bestfit_Fourier(): ')
        print('*get_bestfit_Fourier: x = ',x)
        print('*get_bestfit_Fourier: y = ',y)
        print('*get_bestfit_Fourier: yerr = ',yerr)
    par0 = [np.mean(y),*np.zeros(2*Nterms)]
    if debug:
        print('*get_bestfit_Fourier: par0 = ',par0)

    popt,pcov = curve_fit(
        lambda x,*params:fourier(x,period,Nterms,np.array(params),debug=debug),x,y,sigma=yerr,p0=par0,maxfev=100000)
    if debug:
        print('*get_bestfit_Fourier: optimization finished')
        print('*get_bestfit_Fourier: popt = ',popt)
        print('*get_bestfit_Fourier: pcov = ',pcov)
    if return_yfit:
        y_fit = fourier(x,period,Nterms,popt)
        if debug:
            print('*get_bestfit_Fourier: y_fit = ',y_fit)
        if not return_params:
            return y_fit
        if return_params:
            return y_fit,popt
    elif return_params:
        return popt

def get_chi2_Fourier(x,y,yerr,period,Nterms=4):
    '''
    returns chi square value for the best-fit function at given folding period.
    TODO: copy and paste the content of get_bestfit_Fourier() function
          to make code run faster
    '''
    y_fit = get_bestfit_Fourier(x,y,yerr,period,Nterms,return_yfit=True,return_params=False)
    return np.sum((y-y_fit)**2/yerr**2)/(len(y)-1)
