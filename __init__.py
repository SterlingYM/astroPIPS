'''
-----------
PIPS 0.3.0
---------------------------------------------
Developers: Y. Murakami, A. Hoffman, J.Sunseri, A. Savel
Contact: Yukei Murakami (sterling.astro@berkeley.edu)
License: TBD
---------------------------------------------
Processes photometric data for variable stars.
---------------------------------------------

Classes:
    photdata  --- data container for individual objects and analysis tools
    visualize --- visualization tools for photdata and analysis results
    StellarModels --- various stellar property relationship models (e.g. period-luminosity)

Independent Functions:
    get_bestfit_Fourier(x,y,yerr,period,return_yfit=True,return_params=False)
    get_bestfit_GM(x,y,yerr,period,return_yfit=True,return_params=False)
    get_chi2_Fourier(x,y,yerr,period,Nterms=4)
    get_chi2ref(x,y,yerr)
    Fourier(self,period,params)
    OC(photdata_obj,)
'''
import matplotlib.pyplot as pyplot
import numpy as np
from scipy.optimize import curve_fit
import numba
from multiprocessing import Pool
from functools import partial

#########################
# Independent Functions
#########################
def data_readin_LPP(path,filter='V'):
    '''
    to be updated
    '''
    # takes '.dat' file from LOSS Phot Pypeline (LPP) and returns data in pips.photdata()-readable format.
    # load info
    t,y,y_lower,y_upper = np.loadtxt(path,delimiter='\t',usecols=(0,2,3,4),skiprows=1,unpack=True)
    band                = np.loadtxt(path,delimiter='\t',usecols=6,skiprows=1,dtype=str,unpack=True)

    # uncertainty is not linear in log scale (mag) so (y_upper-y) != (y-y_lower).
    # taking the average of these two is not perfectly accurate, but it works (TODO: modify this?)
    yerr = (y_upper - y_lower)/2 
    
    # separate into specified band
    data = [t[band==filter],y[band==filter],yerr[band==filter]]
    return data

@numba.njit
def Fourier(x,period,Nterms,params,debug=False):
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
        y += C_list[i] * np.cos((i+1)*2*np.pi*x/period + phi_list[i])
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
        lambda x,*params:Fourier(x,period,Nterms,np.array(params),debug=debug),
        x,y,sigma=yerr,p0=par0,maxfev=100000)
    if debug:
        print('*get_bestfit_Fourier: optimization finished')
        print('*get_bestfit_Fourier: popt = ',popt)
        print('*get_bestfit_Fourier: pcov = ',pcov)
    if return_yfit:
        y_fit = Fourier(x,period,Nterms,popt)
        if debug:
            print('*get_bestfit_Fourier: y_fit = ',y_fit)
        if not return_params:
            return y_fit
        if return_params:
            return y_fit,popt
    elif return_params:
        return popt

def get_bestfit_GM(x,y,yerr,period,return_yfit=True,return_params=False):
    '''
    ### Gaussian Mixture Model ###
    returns 
    '''

def get_chi2_Fourier(x,y,yerr,period,Nterms=4):
    '''
    returns chi square value for the best-fit function at given folding period.
    TODO: copy and paste the content of get_bestfit_Fourier() function
          to make code run faster
    '''
    y_fit = get_bestfit_Fourier(x,y,yerr,period,Nterms,return_yfit=True,return_params=False)
    return np.sum((y-y_fit)**2/yerr**2)/(len(y)-1)
    
def get_chi2ref(x,y,yerr):
    '''
    returns non-varying reference of chi2 (model independent)
    '''
    return np.sum((y-np.mean(y))**2/(yerr)**2)/(len(y)-1)

def OC(photdata_obj):
    '''
    TODO: Andrew will write this function
    '''

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

def periodogram_custom(p_min,p_max,N,x,y,yerr,Nterms=1,multiprocessing=True,model='Fourier',**kwargs):
    '''
    model-dependent, individual fitting-based periodogram. Can be customized for any model.
    '''
    MODELS = {
        'Fourier': get_chi2_Fourier
    }
    MODEL_KWARGS = {
        'Fourier': {'x':x,'y':y,'yerr':yerr,'Nterms':Nterms}
    }

    periods = np.linspace(p_min,p_max,N)
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

####################
# Computing helpers
####################
def worker_init(func):
    '''multiprocessing worker initializer for lambda function'''
    global _func
    _func = func

def worker(period):
    '''multiprocessing worker'''
    return _func(period)

class photdata:
    '''
    An object that contains photometric data and analysis results.
    
    variables:
        x(float list): time data
        y(float list): magnitude or flux data
        yerr(float list): error or uncertainty in each y-data
        period(float): detected period of the object. None by default.
        period_err(float): estimated uncertainty in detected period
        amplitude(float): peak-to-peak ydata range (not best-fit function)
        amplitude_err(float): quadrature of yerr for y.max() and y.min()
        label(str): label for the photdata object
        epoch(float): time of maxima, estimated from the datapoint nearest to a maximum
        meanmag: mean magnitude (assuming y-value is in mag)
        p0,p1,p2,...,pN
        A0,A1,A2,...,AN
        
        
    functions (data preparation):
        __init__(self,data,label='')
        
    functions (utilities):
        cut(self,xmin=None,xmax=None,ymin=None,ymax=None,yerr_min=None,yerr_max=None)
        reset_cuts()
        summary()
        
    functions (data processing):
        periodogram(p_min,p_max,N,method,xdata=None,ydata=None,yerr_data=None,plot=False)
        get_period()
        get_period_multi(N,FAR_max=1e-3)
        amplitude_spectrum(p_min,p_max,N,method,plot=False)
        get_bestfit(N,model='Fourier',period=None,plot=True,return_curve=False,return_params=False)
        get_meanmag()
        classify()
    '''
    
    def __init__(self,data,label='',band=''):
        '''
        Takes in a list or numpy array of time-series data
        e.g. ```[time,mag,mag_err]```
        '''
        self.x = data[0]
        self.y = data[1]
        self.yerr = data[2]
        self.period = None
        self.period_err = None
        self.amplitude = None
        self.amplitude_err = None
        self.label = label
        self.band = ''
        self.epoch = None
        self.meanmag = None # based on best-fit function: requires period
        
    ##############
    # utilities
    ##############
    def cut(self,xmin=None,xmax=None,ymin=None,ymax=None,yerr_min=None,yerr_max=None):
        '''
        Cuts data based on given min-max ranges.
        Once this is run, new variables (cut_xmin, cut_xmax, etc.) are assigned
        to save the cut conditions.
        
        The raw (original) data is stored in new variables x_raw,y_raw,yerr_raw.
        If raw variables exist, this function assumes cuts are previously applied,
        and raw variables will not be updated. 
        [i.e. cuts are always applied to the raw data]
        reset_cuts() function resets cuts.
        
        returns nothing.
        '''
        self.cut_xmin = xmin
        self.cut_xmax = xmax
        self.cut_ymin = ymin
        self.cut_ymax = ymax
        self.cut_yerr_min = yerr_min
        self.cut_yerr_max = yerr_max
        self.x_raw = self.x
        self.y_raw = self.y
        self.yerr_raw = self.yerr
        # cut operations here

        
    def reset_cuts(self):
        '''
        resets cuts applied by cut() function.
        '''

    def summary(self):
        '''
        prints out the summary.
        TODO: Jupyter widget?
        '''


    #################
    # analysis tools
    #################      
    def periodogram(self,p_min,p_max,N,method='fast',x=None,y=None,yerr=None,plot=False,
                    multiprocessing=True,Nterms=5,**kwargs):
        '''
        Returns periodogram.
        inputs:
            p_min
            p_max 
            N(int)
            model(str) 
        optional inputs:
            xdata(float list)
            ydata(float list)
            yerr_data(float list)
            plot(bool)
        returns: 
            period
            power 
            (and axis if plot==True)
        '''
        if x==y==yerr==None:
            x = self.x
            y = self.y
            yerr = self.yerr

        METHODS = {
            'fast': periodogram_fast,
            'custom': periodogram_custom
        }
        METHOD_KWARGS = {
            'fast': {'p_min':p_min,'p_max':p_max,'N':N,'x':x,'y':y,'yerr':yerr,'Nterms':Nterms,'multiprocessing':multiprocessing},
            'custom':{'p_min':p_min,'p_max':p_max,'N':N,'x':x,'y':y,'yerr':yerr,'Nterms':Nterms,'multiprocessing':multiprocessing,'model':'Fourier'}
        }
        periods,power = METHODS[method](**METHOD_KWARGS[method])
        return periods, power
    
    def get_period(self,method='Fourier'):
        '''
        
        '''
 
        
    def get_period_multi(self,N,FAR_max=1e-3):
        '''
        multi-period detection. 
        Re-detects P1 and then proceeds to P2, P3, ... PN.
        Pn=None if FAR for nth period exceeds given thershold.
        '''
        
        
    def amplitude_spectrum(self,p_min,p_max,N,model,plot=False):
        '''
        Returns the amplitude spectrum.
        inputs: p_min, p_max, model, plot
        returns: period, amplitude (and axis if plot==True)
        '''
        return period, amplitude

    def get_meanmag(self):
        '''
        calculates an estimated mean magnitude from best-fit curve.
        This method requires a reliable fitting, but is more robust against incomplete sampling in pulsation phase
        '''

    def classify(self):
        '''
        performs the classification of this object based on provided photometric data.
        TODO: this is going to be a big function and requires a lot of work!
        '''
        # self.type = 'RRab'

    def open_widget(self):
        print('in development')

class StellarModels:
    '''
    A supplemental class that provides various stellar property relationships.
    
    (each model is a sub-class that contains unique method functions)
    e.g.
    ~~~~
        >>> from PIPS import photdata, StellarModels
        >>> star = photdata([x,y,yerr])
        >>> model = StellarModels.Hoffman20()
        >>> star_updated = model.calc_Teff(star)
        >>> print(f'{star_updated.Teff:.3e}')
        7.580e3
    ~~~~

    subclasses and method functions for each of them:
        Hoffman20 # <-- this is an example: author of the paper + year is the standard
            calc_color() # <-- This is an example: names don't have to be 'calc_xxxx()'
            calc_Teff()
            calc_luminosity()
        Murakami21
            calc_Teff()
            calc_mass()
            calc_xxx()
        Sunseri22
    '''

class visualize:
    '''
    visualization tool to help statistical analysis.

    variables:
        
    functions:
        to_pandas(photdata_array)
        plot_scatter(df,xcol,ycol,args=None)
        plot_periodogram(photdata_array)
        plot_amplitude_spectrum(df)
        plot_OC(df)
    '''
