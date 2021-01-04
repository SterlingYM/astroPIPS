import numpy as np
from scipy.optimize import curve_fit
from multiprocessing import Pool

def periodogram_fast(p_min,p_max,N,x,y,yerr,Nterms=1,multiprocessing=True,**kwargs):
    '''
    linear algebra-based periodogram.
    '''
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
        '''
        find best-fit solution:
        X*P = Y ==> XT*X*Q = XT*Y
        power(P) = yT*X*
        '''
        # Fourier series prep
        sin_terms = np.sin(ii*2*np.pi*xx/period)/ee
        cos_terms = np.cos(ii*2*np.pi*xx/period)/ee
        X = np.concatenate((sin_terms,cos_terms),axis=1)

        # linear algebra
        XTX = np.dot(X.T,X)
        XTY = np.dot(X.T,Y)
        params = np.linalg.solve(XTX,XTY)
        Yfit = np.dot(X,params)
        return np.dot(Y,Yfit)+np.dot(Y-Yfit,Yfit)
        # return np.dot(XTY.T,params)

    # main
    periods = np.linspace(p_min,p_max,N)
    if multiprocessing:
        pool = Pool()
        chi2 = pool.map(calc_power,periods)
        pool.close()
        pool.join()
    else:
        chi2 = np.asarray(list(map(calc_power,periods)))

    # normalize
    chi2ref = np.dot(Y,Y)
    power = chi2/chi2ref
    return periods,power