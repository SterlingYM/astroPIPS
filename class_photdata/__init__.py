import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import numba
from multiprocessing import Pool
import time

from periodogram.custom import periodogram_custom
from periodogram.linalg import periodogram_fast
from periodogram.models.Fourier import fourier,get_bestfit_Fourier

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
    def periodogram(self,p_min=0.1,p_max=4,custom_periods=None,N=None,method='fast',x=None,y=None,yerr=None,plot=False,multiprocessing=True,Nterms=5,N0=5,model='Fourier',raise_warnings=True,**kwargs):
        '''
        Returns periodogram.
        optional inputs:
            custom_periods(float list): custom list of periods at which periodograms are evaluated at
            xdata(float list)
            ydata(float list)
            yerr_data(float list)
            plot(bool)
            N(int)
            N0(int): number of samples per peak width when N is not specified.
            method(str): 'fast' for multi-term Fourier model or 'custom' for more complicated models
            model(str): model function. This is only effective if method=='custom'.
        returns: 
            period
            power 
            (and axis if plot==True)
        '''
        # prepare data
        if (x is None) and (y is None) and (yerr is None):
            x = self.x
            y = self.y
            yerr = self.yerr
        elif not ((x is not None) and (y is not None) and (yerr is not None)):
            raise ValueError('Input data is incomplete. All x, y, and yerr are needed.')
        
        # determine sampling size
        N_auto = int(N0 * (x.max()-x.min()) * (1/p_min))
        if N==None:
            # VanderPlas 2018 eq. 44
            N = N_auto
        elif (N < N_auto) and raise_warnings:
            print(f'warning: N={N} is smaller than recommended size N={N_auto}')

        # implemented periodogram algorithms
        METHODS = {
            'fast': periodogram_fast,
            'custom': periodogram_custom
        }
        METHOD_KWARGS = {
            'fast': {
                'p_min':p_min,'p_max':p_max,'custom_periods':custom_periods,'N':N,'x':x,'y':y,'yerr':yerr,'Nterms':Nterms,'multiprocessing':multiprocessing
                },
            'custom':{
                'p_min':p_min,'p_max':p_max,'custom_periods':custom_periods,'N':N,'x':x,'y':y,'yerr':yerr,'Nterms':Nterms,'multiprocessing':multiprocessing,'model':model
                }
        }

        # main
        periods,power = METHODS[method](**METHOD_KWARGS[method])
        return periods, power
    
    def get_period(self,p_min=0.1,p_max=4,x=None,y=None,yerr=None,Nterms=5,method='fast',model='Fourier',peaks_to_test=5,N_peak_test=500,debug=False,force_refine=False,default_err=1e-6,**kwargs):
        '''
        detects period.
        '''
        if debug:
            t0 = time.time()
            print(f'{time.time()-t0:.3f}s --- starting the process...')

        # prepare data
        if (x is None) and (y is None) and (yerr is None):
            x = self.x
            y = self.y
            yerr = self.yerr
        elif not ((x is not None) and (y is not None) and (yerr is not None)):
            raise ValueError('Input data is incomplete. All x, y, and yerr are needed.')

        # get periodogram
        period,power = self.periodogram(p_min=p_min,p_max=p_max,x=x,y=y,yerr=yerr,method=method,model=model,Nterms=Nterms,debug=False,**kwargs)

        # select top peaks_to_test independent peaks
        peak_idx = []
        peak_width = 1/(x.max()-x.min())
        peak_idx_width = int(peak_width/(period[1]-period[0]))
        idx_tmp = 0
        sorted_idx = np.flip(power.argsort())
        while len(peak_idx) < peaks_to_test:
            if np.all(abs(sorted_idx[idx_tmp]-peak_idx)>peak_idx_width):
                peak_idx.append(sorted_idx[idx_tmp])
            idx_tmp += 1
        peak_periods = period[peak_idx]

        # perform finer sampling near the peaks
        if debug:
            print(f'{time.time()-t0:.3f}s --- preparing for finer sampling near peaks...')
        custom_periods = np.array([])
        for peak in peak_periods:
            custom_periods = np.concatenate((custom_periods,np.linspace(peak-peak_width,peak+peak_width,N_peak_test)))
        if debug:
            print(f'{time.time()-t0:.3f}s --- performing finer sampling near peaks...')
        period,power = self.periodogram(
            custom_periods=custom_periods,
            x=x,y=y,yerr=yerr,method=method,model=model,Nterms=Nterms,**kwargs
            )
        period = period[power==power.max()][0]
        if debug:
            print(f'{time.time()-t0:.3f}s --- period candidate: ', period)
            
        # model-dependent options
        MODEL_helpers = {
            'Fourier': lambda x,*params: fourier(x,params[0],Nterms,np.array(params[1:]))
        }
        MODEL_bestfit = {
            'Fourier': get_bestfit_Fourier(x,y,yerr,period,Nterms,return_yfit=False,return_params=True)
        }

        # detect scalar multiple of the main pulsation period
        if model=='Fourier':
            if debug: 
                print(f'{time.time()-t0:.3f}s --- detecting scalar multiple of the main pulsation period...')
            factor = np.argmax(abs(MODEL_bestfit[model][1:Nterms]))+1
            if factor != 1:
                period /= factor
                MODEL_bestfit[model] = get_bestfit_Fourier(x,y,yerr,period,Nterms,return_yfit=False,return_params=True)
            if debug:
                print(f'{time.time()-t0:.3f}s --- factor: ',factor)
                print(f'{time.time()-t0:.3f}s --- period: ',period)

        # get uncertainty
        if debug:
            print(f'{time.time()-t0:.3f}s --- estiating the uncertainty...')
        #     print(f'{time.time()-t0:.3f}s --- par0: ',MODEL_bestfit[model])
        popt,pcov = curve_fit(MODEL_helpers[model],x,y,sigma=yerr, p0=[period,*MODEL_bestfit[model]],maxfev=100000)
        period_err = np.sqrt(np.diag(pcov))[0]
        if debug: 
            print(f'{time.time()-t0:.3f}s --- period candidate: ',period)
            print(f'{time.time()-t0:.3f}s --- period fitted: ',popt[0])
            print(f'{time.time()-t0:.3f}s --- period error: ',period_err)
        if period_err == np.inf:
            period_err = default_err
            
        # re-sample if sampling size is not fine enough
        if (period_err < (2*peak_width/N_peak_test)*10) or force_refine:
            if debug:
                print(f'{time.time()-t0:.3f}s --- refining samples...')
            custom_periods = np.linspace(period-period_err,period+period_err,N_peak_test)
            period,power = self.periodogram(
                custom_periods=custom_periods,
                x=x,y=y,yerr=yerr,method=method,model=model,Nterms=Nterms,**kwargs
                )
            period = period[power==power.max()][0]  
            _,pcov = curve_fit(MODEL_helpers[model],x,y,sigma=yerr, p0=[period,*MODEL_bestfit[model][1:]],maxfev=100000)
            period_err = np.sqrt(np.diag(pcov))[0]
            if debug: 
                print(f'{time.time()-t0:.3f}s --- period candidate: ',period)
                print(f'{time.time()-t0:.3f}s --- period fitted: ',popt[0])
                print(f'{time.time()-t0:.3f}s --- period error: ',period_err)

        self.period = period
        self.period_err = period_err
        if debug:
            print(f'{time.time()-t0:.3f}s --- process completed.')
        return period,period_err

        
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

    def plot_lc(self,period=None,invert_yaxis=True,**kwargs):
        '''
        plots phase-folded light curve.
        '''
        if period is None:
            if self.period is None:
                raise ValueError('folding period needs to be specified')
            else:
                period = self.period
        phase = (self.x % period)/period

        # plot
        fig, ax = plt.subplots(1,1,figsize=(8,4))
        if 'color' not in kwargs.keys():
            kwargs['color'] = 'k'
        if 'fmt' not in kwargs.keys():
            kwargs['fmt'] = 'o'
        if 'ms' not in kwargs.keys():
            kwargs['ms'] = 2
        ax.errorbar(phase,self.y,self.yerr,**kwargs)
        ax.errorbar(phase+1,self.y,self.yerr,**kwargs)
        if invert_yaxis:
            ax.invert_yaxis()