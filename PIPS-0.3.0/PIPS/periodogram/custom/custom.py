import numpy as np
from scipy.optimize import curve_fit
import numba
from multiprocessing import Pool

from .models.Fourier import fourier, get_bestfit_Fourier, get_chi2_Fourier
from .models.Gaussian import gaussian, get_bestfit_gaussian, get_chi2_gaussian


### custom model setups: add your functions here
# chi-square calculator
MODELS = {
    'Fourier': get_chi2_Fourier,
    'Gaussian': get_chi2_gaussian
}

def get_chi2ref(x,y,yerr):
    '''
    returns non-varying reference of chi2 (model independent)
    '''
    return np.sum((y-np.mean(y))**2/(yerr)**2)/(len(y)-1)

def periodogram_custom(p_min,p_max,N,x,y,yerr,Nterms=1,multiprocessing=True,model='Fourier',custom_periods=None,**kwargs):
    '''
    model-dependent, individual fitting-based periodogram. Can be customized for any model.
    '''
    # kwargs for chi-square calculators
    MODEL_KWARGS = {
        'Fourier': {'x':x,'y':y,'yerr':yerr,'Nterms':Nterms},
        'Gaussian': {'x':x,'y':y,'yerr':yerr,'Nterms':Nterms}
    }

    # prepare periods
    if (p_min is not None) and (p_max is not None):
        periods = np.linspace(p_min,p_max,N)
    elif custom_periods is not None:
        periods = custom_periods
    else:
        raise ValueError('period range or period list are not given')

    # main
    if multiprocessing==True:
        global mp_worker
        def mp_worker(period):
            return MODELS[model](period=period,**MODEL_KWARGS[model])
        pool = Pool()
        chi2 = pool.map(mp_worker,periods)
        pool.close()
        pool.join()
    else:
        chi2 = np.array(list(map(lambda period: MODELS[model](period=period,**MODEL_KWARGS[model]),periods)))
    chi2ref = get_chi2ref(x,y,yerr)
    power = 1 - chi2/chi2ref

    return periods, power