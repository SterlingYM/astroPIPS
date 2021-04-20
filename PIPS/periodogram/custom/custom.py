import numpy as np
from scipy.optimize import curve_fit
import numba
from multiprocessing import Pool

from .models.Fourier import fourier, fourier_p0
from .models.Gaussian import gaussian, gaussian_p0

### custom model setups: add your functions here
# chi-square calculator
MODELS = {
    'Fourier': fourier,
    'Gaussian': gaussian,
    # 'Super-Gaussian': super_gaussian,
}
p0_funcs = {
    'Fourier': fourier_p0,
    'Gaussian': gaussian_p0,
}

def prepare_MODEL_KWARGS(model,x,y,yerr,**kwargs):
    KWARGS = {}
    # pre-inplemented model-dependent kwargs
    if model=='Fourier':
        requirements = ['Nterms']
    if model=='Gaussian':
        requirements = []

    # prepare KWARGS
    for key in requirements:
        if key in kwargs:
            KWARGS[key]=kwargs[key]
        else:
            raise ValueError(model+f' function requires the following argumnets: {requirements}')
    return KWARGS

def get_bestfit(MODEL,p0_func,x,y,yerr,period,return_yfit=True,return_params=False,maxfev=1000,**kwargs):
    p0 = p0_func(x,y,yerr,period,**kwargs)
    try:
        popt,pcov = curve_fit(
            lambda x,*params:MODEL(x,period,np.array(params),**kwargs),x,y,sigma=yerr,p0=p0,maxfev=maxfev)
        y_fit = MODEL(x,period,np.array(popt),**kwargs)
    except RuntimeError:
        y_fit = np.ones_like(y) * np.mean(y) # equiv. to zero in periodogram
    if return_yfit:
        if not return_params:
            return y_fit
        if return_params:
            return y_fit,popt
    elif return_params:
        return popt

def get_chi2(MODEL,p0_func,x,y,yerr,period,**kwargs):
    '''
    returns chi square value for the best-fit function at given folding period.
    '''
    y_fit = get_bestfit(MODEL,p0_func,x,y,yerr,period,return_yfit=True,return_params=False,**kwargs)
    return np.sum((y-y_fit)**2/yerr**2)

def get_chi2ref(x,y,yerr):
    '''
    returns non-varying reference of chi2 (model independent)
    '''
    return np.sum((y-np.mean(y))**2/(yerr)**2)

def periodogram_custom(x,y,yerr,p_min=None,p_max=None,N=None,p0_func=None,multiprocessing=True,model='Fourier',custom_periods=None,**kwargs):
    '''
    model-dependent, individual fitting-based periodogram. Can be customized for any model.
    '''
    # select models
    if isinstance(model, str):
        MODEL = MODELS[model]
        KWARGS = {**kwargs,**prepare_MODEL_KWARGS(model,x,y,yerr,**kwargs)}
        P0_FUNC = p0_funcs[model]
    elif hasattr(model, '__call__'):
        MODEL = model
        KWARGS= kwargs
        if hasattr(p0_func, '__call__'):
            P0_FUNC = p0_func
        else:
            raise ValueError('custom model requires initial-guess prep function (p0_func).')
    else:
        raise ValueError('model has to be either a function or a pre-defined function name')

    # prepare periods
    if (p_min is not None) and (p_max is not None):
        periods = np.linspace(p_min,p_max,N)
    elif custom_periods is not None:
        periods = custom_periods
    else:
        raise ValueError('period range or period list are not given')

    # main
    global mp_worker
    def mp_worker(period):
        return get_chi2(MODEL=MODEL,p0_func=P0_FUNC,x=x,y=y,yerr=yerr,period=period,**KWARGS)

    if multiprocessing==True:
        pool = Pool()
        chi2 = pool.map(mp_worker,periods)
        pool.close()
        pool.join()
    else:
        chi2 = np.array(list(map(mp_worker,periods)))
    chi2ref = get_chi2ref(x,y,yerr)
    power = 1 - chi2/chi2ref

    return periods, power