import matplotlib.pyplot as pyplot
import numpy as np
from scipy.optimize import curve_fit
import numba
from multiprocessing import Pool

from periodogram.custom import periodogram_custom
from periodogram.linalg import periodogram_fast

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
        # return period, amplitude

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