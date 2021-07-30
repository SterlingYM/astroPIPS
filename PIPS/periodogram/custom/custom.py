import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import numba
from multiprocessing import Pool
import warnings
warnings.simplefilter("ignore", OptimizeWarning)

from .models.Fourier import fourier, fourier_p0
from .models.Gaussian import gaussian, gaussian_p0

######### custom models: add your functions here ##########
MODELS = {
    'Fourier': fourier,
    'Gaussian': gaussian,
}
P0_FUNCS = {
    'Fourier': fourier_p0,
    'Gaussian': gaussian_p0,
}
###########################################################

def check_MODEL_KWARGS(model,kwarg_for_helper=True,**kwargs):
    KWARGS = {}
    if kwarg_for_helper:
        helper_kwargs = ['return_yfit','return_params','return_pcov','fit_period','maxfev']
    else:
        helper_kwargs = []
    # pre-inplemented model-dependent kwargs
    if model=='Fourier':
        requirements = ['Nterms']
    elif model=='Gaussian':
        requirements = ['Nterms','p']
    else:
        requirements = kwargs.keys()

    # prepare KWARGS
    for key in [*requirements,*helper_kwargs]:
        if key in kwargs:
            KWARGS[key]=kwargs[key]
        # else:
        #     raise ValueError(model+f' function requires the following argumnets: {requirements}')
    return KWARGS

def get_bestfit(MODEL,p0_func,x,y,yerr,period,return_yfit=True,return_params=False,return_pcov=False,fit_period=False,maxfev=1000,**kwargs):
    if fit_period:
        p0 = [period,*p0_func(x,y,yerr,period,**kwargs)]
        try:
            popt,pcov = curve_fit(
                lambda x,period,*params:MODEL(x,period,np.array(params),**kwargs),x,y,sigma=yerr,p0=p0,maxfev=maxfev)
            y_fit = MODEL(x,period,np.array(popt),**kwargs)
        except RuntimeError:
            y_fit = np.ones_like(y) * np.mean(y) # equiv. to zero in periodogram
        except TypeError:
            print('received incorrect kwargs: ',kwargs)
    else:
        p0 = p0_func(x,y,yerr,period,**kwargs)
        try:
            popt,pcov = curve_fit(
                lambda x,*params:MODEL(x,period,np.array(params),**kwargs),x,y,sigma=yerr,p0=p0,maxfev=maxfev)
            y_fit = MODEL(x,period,np.array(popt),**kwargs)
        except RuntimeError:
            y_fit = np.ones_like(y) * np.mean(y) # equiv. to zero in periodogram
        except TypeError:
            print('received incorrect kwargs: ',kwargs)
    if return_yfit:
        if not return_params:
            return y_fit
        if return_params:
            return y_fit,popt
    elif return_params and return_pcov:
        return popt,pcov
    elif return_params:
        return popt
    elif return_pcov:
        return pcov

def get_chi2(MODEL,p0_func,x,y,yerr,period,**kwargs):
    '''
    returns chi square value for the best-fit function at given folding period.
    '''
    y_fit = get_bestfit(MODEL,p0_func,x,y,yerr,period,return_yfit=True,return_params=False,**kwargs)
    return np.sum((y-y_fit)**2/yerr**2)

def get_likelihood(MODEL,p0_func,x,y,yerr,period,**kwargs):
    '''
    returns Gaussian likelihood for the best-fit function at given folding period.
    '''
    y_fit = get_bestfit(MODEL,p0_func,x,y,yerr,period,return_yfit=True,return_params=False,**kwargs)
    lik = np.prod(
        np.exp(-0.5*(y-y_fit)**2/(yerr**2)) / (np.sqrt(2*np.pi)*yerr)
        )
    return lik

def get_loglik(MODEL,p0_func,x,y,yerr,period,**kwargs):
    '''
    returns Gaussian likelihood for the best-fit function at given folding period.
    '''
    y_fit = get_bestfit(MODEL,p0_func,x,y,yerr,period,return_yfit=True,return_params=False,**kwargs)
    lik = -0.5 * np.sum(
        (y-y_fit)**2/(yerr**2) + np.log(2*np.pi*yerr**2)
        )
    return lik

def get_chi2ref(x,y,yerr):
    '''
    returns non-varying reference of chi2 (model independent)
    '''
    return np.sum((y-np.mean(y))**2/(yerr)**2)

def periodogram_custom(x,y,yerr,p_min=None,p_max=None,N=None,p0_func=None,multiprocessing=True,model='Fourier',custom_periods=None,repr_mode='chisq',**kwargs):
    '''
    model-dependent, individual fitting-based periodogram. Can be customized for any model.
    '''
    # select models
    if isinstance(model, str):
        MODEL = MODELS[model]
        KWARGS = {**check_MODEL_KWARGS(model,**kwargs)}
        P0_FUNC = P0_FUNCS[model]
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
    if custom_periods is not None:
        periods = custom_periods
    elif (p_min is not None) and (p_max is not None):
        # periods = np.linspace(p_min,p_max,N)
        periods = 1/np.linspace(1/p_max,1/p_min,N)
    else:
        raise ValueError('period range or period list are not given')

    # main
    REPRs = {
        'chisq':get_chi2,
        'likelihood':get_likelihood,
        'log-likelihood':get_loglik
    }
    global mp_worker
    def mp_worker(period):
        return REPRs[repr_mode](MODEL=MODEL,p0_func=P0_FUNC,x=x,y=y,yerr=yerr,period=period,**KWARGS)

    if multiprocessing==True:
        pool = Pool()
        chi2 = pool.map(mp_worker,periods)
        pool.close()
        pool.join()
    else:
        chi2 = np.array(list(map(mp_worker,periods)))
    
    if repr_mode=='likelihood' or repr_mode=='log-likelihood':
        return periods,chi2

    chi2ref = get_chi2ref(x,y,yerr)
    power = 1 - chi2/chi2ref

    return periods, power