import numpy as np
from scipy.optimize import curve_fit
from multiprocessing import Pool

def periodogram_fast(p_min,p_max,N,x,y,yerr,Nterms=1,multiprocessing=True,custom_periods=None,model='Fourier',repr_mode='chisq',normalize=True,**kwargs):
    ''' Generates the periodogram based on linear algebra method.
    
    Args:
        p_min: the minimum period of the periodogram grid.
        p_max: the maximum period of the periodogram grid.
        N: the number of samples in the grid.
        x: the time data.
        y: the mag/flux data.
        yerr: the uncertainties in mag/flux data.
        Nterms (int): the number of terms in Fourier/Gaussian models.
        multiprocessing (bool): the option to use multiprocessing feature.
        custom_periods: the user-defined period values at which the periodogram is evaluated.
        model (str): the lightcurve model. It has to be the name of the pre-implemented method ('Fourier' or 'Gaussian').
        repr_mode (str)['likelihood','lik','log-likelihood','loglik','chi-square','chisq']: the periodogram representation.
        normalize (bool): an option to normalize the resulting periodogram.
        
    Returns:
        periods: periods at which periodogram is evaluated.
        power: periodogram values.
    
    '''
    # avoid invalid log for Gaussian model
    if model=='Gaussian':
        y -= np.min(y)-1e-10
        yerr = yerr/y
        y = np.log(y)

    # weighted y prep
    w = (1/yerr)**2 / np.sum((1/yerr)**2)
    Y = (y - np.dot(w,y))/yerr # w*y = weighted mean

    # matrix prep
    ii = (np.arange(Nterms)+1).repeat(len(x)).reshape(Nterms,len(x),).T
    xx = x.repeat(Nterms).reshape(len(x),Nterms)
    ee = yerr.repeat(Nterms).reshape(len(x),Nterms)
    if xx.shape != ee.shape:
        raise ValueError('y-error data size does not match x-data size')

    # worker prep -- calculate power (= chi2ref-chi2)
    global calc_power
    def calc_power(period):
        ''' find best-fit solution.
        
        X*P = Y ==> XT*X*Q = XT*Y
        power(P) = yT*X*
        
        Args:
            period (float): the phase-folding period.
            
        Returns:
            power (numpy array): the periodogram value at given period.
        '''

        if model == 'Fourier':
            # Fourier series prep
            const_term = np.ones_like(x).reshape(len(x),1)
            sin_terms = np.sin(ii*2*np.pi*xx/period)/ee
            cos_terms = np.cos(ii*2*np.pi*xx/period)/ee
            X = np.concatenate((const_term,sin_terms,cos_terms),axis=1)
        elif model == 'Gaussian':
            # Gaussian series prep
            const_term = np.tile(np.ones_like(x).reshape(len(x),1),(1,Nterms))
            linear_terms = (xx%period)/ee
            square_terms = (xx%period)**2/ee
            X = np.concatenate((const_term,linear_terms,square_terms),axis=1)

        # linear algebra
        XTX = np.dot(X.T,X)
        XTY = np.dot(X.T,Y)
        params = np.linalg.solve(XTX,XTY)
        Yfit = np.dot(X,params)
        if repr_mode == 'chisq':
            return np.dot(Y,Yfit)+np.dot(Y-Yfit,Yfit)
        elif repr_mode in ['likelihood','lik']:
            loglik = -0.5*np.sum((Y-Yfit)**2 + np.log(2*np.pi*yerr**2))
            return loglik # note: this needs to be brought back to exp later
            # return np.prod(np.exp(-0.5*(Y-Yfit)**2)/(np.sqrt(2*np.pi)*yerr))
        elif repr_mode in ['log-likelihood','loglik']:
            return -0.5*np.sum((Y-Yfit)**2 + np.log(2*np.pi*yerr**2))

    # period prep
    if custom_periods is not None:
        periods = custom_periods
    elif (p_min is not None) and (p_max is not None):
        # periods = np.linspace(p_min,p_max,N)
        periods = 1/np.linspace(1/p_max,1/p_min,N)
    else:
        raise ValueError('period range or period list are not given')

    # main
    if multiprocessing:
        pool = Pool()
        chi2 = pool.map(calc_power,periods)
        pool.close()
        pool.join()
    else:
        chi2 = np.asarray(list(map(calc_power,periods)))

    # normalize
    if repr_mode in ['log-likelihood','loglik']:
        if normalize:
            return periods,chi2-chi2.max()
        else:
            return periods,chi2
    elif repr_mode in ['likelihood','lik']:
        if normalize:
            return periods,np.exp(chi2-chi2.max())
        else:
            return periods,np.exp(chi2)
    else:
        chi2ref = np.dot(Y,Y)
        power = chi2/chi2ref
        return periods,power