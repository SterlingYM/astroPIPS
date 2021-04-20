import numpy as np
from scipy.optimize import curve_fit
import numba
from multiprocessing import Pool

@numba.njit
def gaussian(x,period,params,Nterms=1,p=1):
    '''
    A Gaussian function (model) that calculates y-value 
    at each x-value for given period and parametrs.
    ** IMPORTANT NOTE: ```params``` has to be a numpy array (not python list)
    '''
    # if debug:
    #     print('*Gaussian: starting Gaussian')
    #     print('*Gaussian: params = ',params)
    y = np.ones_like(x) * params[0]
    p_list = params[1:Nterms+1]
    mu_list = params[Nterms+1:2*Nterms+1]
    sigma_list = params[2*Nterms+1:]
    # if debug:
    #     print('*Gaussian: y_initial = ',y)
    #     print('*Gaussian: A_list = ',p_list)
    #     print('*Gaussian: phi_list = ',mu_list)
    #     print('*Gaussian: sigma_list = ',sigma_list)

    # prepare phase-folded x-values
    mod = np.remainder(x,period)
    for i in range(Nterms):
        y = y + p_list[i] * np.exp(-(0.5*(mod-mu_list[i])**2/sigma_list[i]**2)**p)
    # if debug:
    #     print('*Gaussian: y after calculation = ',y)
    return y

def gaussian_p0(x,y,yerr,period,Nterms=1,**kwargs):
    return [np.mean(y),*np.full(Nterms,1),*np.full(Nterms,period/2),*np.full(Nterms,period/4)]

def get_bestfit_gaussian(x,y,yerr,period,Nterms,return_yfit=True,return_params=False,debug=False):
    '''
    ### Gaussian Model ###
    returns best-fit y-values at given x
    if return_yfit==True, it returns best-fit y-values at given x
    if return_params==True, it returns best-fit parameters (model-dependent)
    NOTE: Gaussian parameters are not bound to keep the code fast.
    For stellar parameter calculation purpose, use tools in StellarModels class.
    '''
    if debug:
        print('*get_bestfit_gaussian: starting process get_bestfit_gaussian(): ')
        print('*get_bestfit_gaussian: x = ',x)
        print('*get_bestfit_gaussian: y = ',y)
        print('*get_bestfit_gaussian: yerr = ',yerr)
    par0 = [np.mean(y),*np.full(Nterms,1),*np.full(Nterms,period/2),*np.full(Nterms,period/4)]
    if debug:
        print('*get_bestfit_gaussian: par0 = ',par0)

    # TODO: is this the best solution?
    try:
        popt,pcov = curve_fit(
            lambda x,*params:gaussian(x,period,Nterms,np.array(params),debug=debug),x,y,sigma=yerr,p0=par0,maxfev=100000)
    except:
        popt = np.array([y.mean(),*np.full(Nterms*2,0.0),*np.full(Nterms,1.0)])
    if debug:
        print('*get_bestfit_gaussian: optimization finished')
        print('*get_bestfit_gaussian: popt = ',popt)
        print('*get_bestfit_gaussian: pcov = ',pcov)
    if return_yfit:
        y_fit = gaussian(x,period,Nterms,popt)
        if debug:
            print('*get_bestfit_gaussian: y_fit = ',y_fit)
        if not return_params:
            return y_fit
        if return_params:
            return y_fit,popt
    elif return_params:
        return popt

# def get_chi2_gaussian(x,y,yerr,period,Nterms=4):
#     '''
#     returns chi square value for the best-fit function at given folding period.
#     '''
#     y_fit = get_bestfit_gaussian(x,y,yerr,period,Nterms,return_yfit=True,return_params=False)
#     return np.sum((y-y_fit)**2/yerr**2)/(len(y)-1)
