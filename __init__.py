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


#########################
# Independent Functions
#########################

def get_bestfit_Fourier(x,y,yerr,period,return_yfit=True,return_params=False):
    '''
    ### Fourier Model ###
    returns best-fit y-values at given x
    if return_yfit==True, it returns best-fit y-values at given x
    if return_params==True, it returns best-fit parameters (model-dependent)
    '''
    return y-fit

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
    y_fit = get_bestfit_Fourier(x,y,yerr,period,return_yfit=True,return_params=False)
    return np.sum((y-y_fit)**2/yerr**2)/(len(y)-1)
    
def get_chi2ref(x,y,yerr):
    '''
    returns non-varying reference of chi2 (model independent)
    '''

def Fourier(self,period,params):
    '''
    A Fourier function (model) that calculates y-value 
    at each x-value for given period and parametrs.
    '''
    return y

def OC(photdata_obj,)
    '''
    TODO: Andrew will write this function
    '''

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

        
    def reset_cuts():
        '''
        resets cuts applied by cut() function.
        '''

    def summary():
        '''
        prints out the summary.
        TODO: Jupyter widget?
        '''
    
    #################
    # analysis tools
    #################      
        
    def periodogram(p_min,p_max,N,model,xdata=None,ydata=None,yerr_data=None,plot=False,
                    multiprocessing=True):
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
        if xdata==ydata==yerr_data==None:
            xdata = self.x
            ydata = self.y
            yerr_data = self.yerr
        periods = np.linspace(p_min,p_max,N)
        
        ## model-dependent statements here
        chi2 = pool.map(get_chi2_Fourier,periods,args=[model])
        return periods, powers
    
    def get_period(method='Fourier'):
        '''
        
        '''
        
    def get_period_multi(N,FAR_max=1e-3):
        '''
        multi-period detection. 
        Re-detects P1 and then proceeds to P2, P3, ... PN.
        Pn=None if FAR for nth period exceeds given thershold.
        '''
        
        
    def amplitude_spectrum(p_min,p_max,N,model,plot=False):
        '''
        Returns the amplitude spectrum.
        inputs: p_min, p_max, model, plot
        returns: period, amplitude (and axis if plot==True)
        '''
        return period, amplitude

    def get_meanmag()
        '''
        calculates an estimated mean magnitude from best-fit curve.
        This method requires a reliable fitting, but is more robust against incomplete sampling in pulsation phase
        '''

    def classify()
        '''
        performs the classification of this object based on provided photometric data.
        TODO: this is going to be a big function and requires a lot of work!
        '''
        # self.type = 'RRab'

    def open_widget()

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
    
