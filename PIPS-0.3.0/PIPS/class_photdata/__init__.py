import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import numba
from multiprocessing import Pool
import time

from ..periodogram.custom import periodogram_custom
from ..periodogram.linalg import periodogram_fast
from ..periodogram.custom.models.Fourier import fourier,get_bestfit_Fourier
from ..periodogram.custom.models.Gaussian import gaussian,get_bestfit_gaussian

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
        prepare_data()
        get_bestfit_curve(self,x=None,y=None,yerr=None,period=None,model='Fourier',Nterms=5,x_th=None)
        get_bestfit_amplitude(self,x=None,y=None,yerr=None,period=None,model='Fourier',Nterms=5)
        get_meanmag(self,x=None,y=None,yerr=None,period=None,model='Fourier',Nterms=5)

    functions (data processing):
        periodogram(self,p_min=0.1,p_max=4,custom_periods=None,N=None,method='fast',x=None,y=None,yerr=None,plot=False,multiprocessing=True,Nterms=5,N0=5,model='Fourier',raise_warnings=True,**kwargs)
        get_period(self,p_min=0.1,p_max=4,x=None,y=None,yerr=None,Nterms=5,method='fast',model='Fourier',peaks_to_test=5,N_peak_test=500,debug=False,force_refine=False,default_err=1e-6,**kwargs)
        get_period_multi(self,N,FAR_max=1e-3,model='Fourier',Nterms=5,**kwargs)
        amplitude_spectrum(self,p_min,p_max,N,model='Fourier',grid=10000,plot=False,Nterms=5,**kwargs)
        get_bestfit(N,model='Fourier',period=None,plot=True,return_curve=False,return_params=False)
        classify(self)
        open_widget(self)
        plot_lc(self,period=None,invert_yaxis=True,**kwargs)
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
        self.epoch_offset = None
        self.meanmag = None # based on best-fit function: requires period
        
    ##############
    # utilities
    ##############
    def check_model(self, input_model, model_dict):
        """
        Checks that a given input model is available.
        input_model : (str) user-input model.
        model_dict : (dict) dictionary containing model strings as keys and arbitrary functions as values.
        """
        if input_model not in model_dict.keys():
            raise ValueError("""Input model is not available. Currently available models \
                                include 'Gaussian' and 'Fourier'.""")
        
    def cut(self,xmin=None,xmax=None,ymin=None,ymax=None,yerr_min=None,yerr_max=None):
        '''
        Cuts data based on given min-max ranges.
        Once this is run, new variables (cut_xmin, cut_xmax, etc.) are assigned to save the cut conditions.
        
        The raw (original) data is stored in new variables x_raw,y_raw,yerr_raw. If raw variables exist, this function assumes cuts are previously applied, and raw variables will not be updated. 
        [i.e. cuts are always applied to the raw data]
        reset_cuts() function resets cuts.
        
        returns nothing.
        '''
        # first-time operation
        if not hasattr(self,'cut_xmin'):
            # cut_xmin does not exist until cut() is run for the first time. Once it is run, cut_xmin==None and does exist even if the cut is not applied in x.
            self.x_raw = self.x
            self.y_raw = self.y
            self.yerr_raw = self.yerr

            # initialize 
            self.cut_xmin = xmin
            self.cut_xmax = xmax
            self.cut_ymin = ymin
            self.cut_ymax = ymax
            self.cut_yerr_min = yerr_min
            self.cut_yerr_max = yerr_max

        # second time and after: update cuts
        else:
            if xmin is not None:
                self.cut_xmin = xmin
            if xmax is not None:
                self.cut_xmax = xmax
            if ymin is not None:
                self.cut_ymin = ymin
            if ymax is not None:
                self.cut_ymax = ymax
            if yerr_min is not None:
                self.cut_yerr_min = yerr_min
            if yerr_max is not None:
                self.cut_yerr_max = yerr_max

        # prepare cut conditions
        condition = np.full(self.x_raw.shape, True, dtype=bool)
        if self.cut_xmin is not None:
            condition = condition & (self.x_raw >= self.cut_xmin)
        if self.cut_xmax is not None:
            condition = condition & (self.x_raw <= self.cut_xmax)
        if self.cut_ymin is not None:
            condition = condition & (self.y_raw >= self.cut_ymin)
        if self.cut_ymax is not None:
            condition = condition & (self.y_raw <= self.cut_ymax)
        if self.cut_yerr_min is not None:
            condition = condition & (self.yerr_raw >= self.cut_yerr_min)
        if self.cut_yerr_max is not None:
            condition = condition & (self.yerr_raw <= self.cut_yerr_max)

        # apply cuts
        self.x = self.x_raw[condition]
        self.y = self.y_raw[condition]
        self.yerr = self.yerr_raw[condition]
        
    def reset_cuts(self):
        '''
        resets cuts applied by cut() function.
        '''
        if hasattr(self,'x_raw'):
            self.cut_xmin = None
            self.cut_xmax = None
            self.cut_ymin = None
            self.cut_ymax = None
            self.cut_yerr_min = None
            self.cut_yerr_max = None
            self.x = self.x_raw
            self.y = self.y_raw
            self.yerr = self.yerr_raw

    def summary(self):
        '''
        prints out the summary.
        TODO: Jupyter widget?
        '''
        raise NotImplementedError

    def prepare_data(self,x,y,yerr):
        if (x is None) and (y is None) and (yerr is None):
            x = self.x
            y = self.y
            yerr = self.yerr
        elif not ((x is not None) and (y is not None) and (yerr is not None)):
            raise ValueError('Input data is incomplete. All x, y, and yerr are needed.')
        return x,y,yerr

    def get_bestfit_curve(self,x=None,y=None,yerr=None,period=None,model='Fourier',Nterms=5,x_th=None,**kwargs):
        '''
        Calculates the best-fit smooth curve.
        '''
        # prepare data
        x,y,yerr = self.prepare_data(x,y,yerr)
        
        # use automatically determined period if period is not explicitly given
        if period == None:
            if self.period == None:
                period, _ = self.get_period(**kwargs)
            period = self.period

        # model-dependent options
        MODEL_bestfit = {
            'Fourier': get_bestfit_Fourier,
            'Gaussian': get_bestfit_gaussian
        }
        MODELS = {
            'Fourier': fourier,
            'Gaussian': gaussian
        }
        self.check_model(model, MODELS)
        # 
        popt = MODEL_bestfit[model](x,y,yerr,period,Nterms,return_yfit=False,return_params=True)
        
        # construct theoretical curve
        if x_th is None:
            x_th = np.linspace(0,period,1000)
        y_th = MODELS[model](x_th,period,Nterms,np.array(popt))

        return x_th,y_th

    def get_bestfit_amplitude(self,x=None,y=None,yerr=None,period=None,model='Fourier',Nterms=5):
        '''
        calculates the amplitude of best-fit curve.
        '''
        _,y_th = self.get_bestfit_curve(x,y,yerr,period,model,Nterms)
        return np.max(y_th)-np.min(y_th)

    def get_meanmag(self,x=None,y=None,yerr=None,period=None,model='Fourier',Nterms=5):
        '''
        calculates an estimated mean magnitude from best-fit curve.
        This method requires a reliable fitting, but is more robust against incomplete sampling in pulsation phase
        '''
        _,y_th = self.get_bestfit_curve(x,y,yerr,period,model,Nterms)
        return np.mean(y_th)

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
        x,y,yerr = self.prepare_data(x,y,yerr)
        
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
        # model-dependent options
        MODEL_helpers = {
            'Fourier': lambda x,*params: fourier(x,params[0],Nterms,np.array(params[1:])),
            'Gaussian': lambda x,*params: gaussian(x,params[0],Nterms,np.array(params[1:])),
        }
        MODEL_bestfit = {
            'Fourier': get_bestfit_Fourier,
            'Gaussian': get_bestfit_gaussian
        }
        
        self.check_model(model, MODEL_bestfit)

        # debug mode option outputs the progress 
        # (TODO: change this to verbosity - or logger?)
        if debug:
            t0 = time.time()
            print(f'{time.time()-t0:.3f}s --- starting the process...')
            print(f'{time.time()-t0:.3f}s --- preparing data...')

        # prepare data
        x,y,yerr = self.prepare_data(x,y,yerr)

        # get periodogram
        if debug:
            print(f'{time.time()-t0:.3f}s --- getting a periodogram...')
        period,power = self.periodogram(p_min=p_min,p_max=p_max,x=x,y=y,yerr=yerr,method=method,model=model,Nterms=Nterms,debug=False,**kwargs)

        # select top peaks_to_test independent peaks
        if debug:
            print(f'{time.time()-t0:.3f}s --- detecting top {peaks_to_test} peaks...')
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
            
        model_bestfit = MODEL_bestfit[model](x,y,yerr,period,Nterms,return_yfit=False,return_params=True)

        # detect aliasing
        if model=='Fourier':
            if debug: 
                print(f'{time.time()-t0:.3f}s --- detecting aliasing...')
            factor = np.argmax(abs(model_bestfit[1:Nterms]))+1
            if factor != 1:
                period /= factor
                model_bestfit = MODEL_bestfit[model](x,y,yerr,period,Nterms,return_yfit=False,return_params=True)
            if debug:
                print(f'{time.time()-t0:.3f}s --- alias factor: ',factor)
                print(f'{time.time()-t0:.3f}s --- period candidate: ',period)

        # get uncertainty
        if debug:
            print(f'{time.time()-t0:.3f}s --- estimating the uncertainty...')
        popt,pcov = curve_fit(MODEL_helpers[model],x,y,sigma=yerr, p0=[period,*model_bestfit],maxfev=100000)
        period_err = np.sqrt(np.diag(pcov))[0]
        if debug: 
            print(f'{time.time()-t0:.3f}s --- period candidate: ',period)
            print(f'{time.time()-t0:.3f}s --- period fitted*: ',popt[0])
            print(f'{time.time()-t0:.3f}s --- period error: ',period_err)
        if period_err == np.inf:
            # automatically activate the refinement process
            period_err = 0 

        # re-sample if sampling size is not fine enough
        if (period_err < (2*peak_width/N_peak_test)*10) or force_refine:
            if debug:
                print(f'{time.time()-t0:.3f}s --- refining samples...')
                print(f'{time.time()-t0:.3f}s --- refining search width = {peak_width/10:.3e}')

            # prepare new search width -- narrower and thus refined
            #TODO: discuss this method
            new_search_width = peak_width/N_peak_test*100 
            custom_periods = np.linspace(period-new_search_width,period+new_search_width,N_peak_test)

            # get periodogram
            period,power = self.periodogram(
                custom_periods=custom_periods,
                x=x,y=y,yerr=yerr,method=method,model=model,Nterms=Nterms,**kwargs
                )
            period = period[power==power.max()][0]

            # get uncertainty
            model_bestfit = MODEL_bestfit[model](x,y,yerr,period,Nterms,return_yfit=False,return_params=True)
            _,pcov = curve_fit(MODEL_helpers[model],x,y,sigma=yerr, p0=[period,*model_bestfit],maxfev=100000)
            period_err = np.sqrt(np.diag(pcov))[0]
            if debug: 
                print(f'{time.time()-t0:.3f}s --- period candidate: ',period)
                print(f'{time.time()-t0:.3f}s --- period fitted*: ',popt[0])
                print(f'{time.time()-t0:.3f}s --- period error: ',period_err)

        # check: is the size of uncertainty close to the deviation size
        # within a factor of two or less?
        fit_peak_deviation = abs(popt[0]-period)
        if debug:
            print(f'{time.time()-t0:.3f}s --- * validating period error...')
            print(f'{time.time()-t0:.3f}s --- * fitted period - peak period = {fit_peak_deviation:.2e}')
            print(f'{time.time()-t0:.3f}s --- * expected deviation size = {period_err:.2e}')
        if (fit_peak_deviation > 2*period_err) or (period_err==np.inf):
            warningMessage = 'warning: provided uncertainty may not be accurate. Try increasing sampling size (N_peak_test, default 500) and/or turn on the force_refine option.'
            print(warningMessage)
        elif debug:
            print(f'{time.time()-t0:.3f}s --- * period error validated')

        if period_err == np.inf:
            print('warning: error size infinity: replacing with periodogram peak width')
            period_err = peak_width

        # finalize
        self.period = period
        self.period_err = period_err
        if debug:
            print(f'{time.time()-t0:.3f}s ---','period = {:.{precision}f} +- {:.{precision}f}d'.format(period,period_err,precision=5 if period_err==np.inf else int(abs(np.log10(period_err))+2)))
            print(f'{time.time()-t0:.3f}s --- process completed.')
        return period,period_err
   
    def get_period_multi(self,N,FAR_max=1e-3,model='Fourier',Nterms=5,**kwargs):
        '''
        multi-period detection. 
        Re-detects P1 and then proceeds to P2, P3, ... PN.
        Pn=None if FAR for nth period exceeds given thershold.
        '''
        # TODO: implement FAR

        # model-dependent options
        MODEL_bestfit = {
            'Fourier': get_bestfit_Fourier,
            'Gaussian': get_bestfit_gaussian
        }
        self.check_model(model, MODEL_bestfit)
        
        # data prep
        x_prewhitened = self.x.copy()
        y_prewhitened = self.y.copy()
        yerr_prewhitened = self.yerr.copy()

        # repeats period detection -> prewhitening
        periods = []
        period_errors = []
        amplitudes = []

        for _ in range(N):
            period,period_err = self.get_period(
                x=x_prewhitened,
                y=y_prewhitened,
                yerr=yerr_prewhitened,
                model=model,Nterms=Nterms,**kwargs)
            periods.append(period)
            period_errors.append(period_err)
            amp = self.get_bestfit_amplitude(
                x=x_prewhitened,
                y=y_prewhitened,
                yerr=yerr_prewhitened,
                period=period,
                model=model,Nterms=Nterms)
            amplitudes.append(amp)
            y_prewhitened -= MODEL_bestfit[model](
                x_prewhitened,
                y_prewhitened,
                yerr_prewhitened,
                period,
                Nterms,return_yfit=True,return_params=False)
        return periods,period_errors,amplitudes

    def amplitude_spectrum(self,p_min,p_max,N,model='Fourier',grid=10000,plot=False,Nterms=5,**kwargs):
        '''
        Returns the amplitude spectrum.
        inputs: p_min, p_max, model, plot
        returns: period, amplitude (and axis if plot==True)
        '''

        periods,period_errors,amplitudes = self.get_period_multi(
            N,
            p_min=p_min,
            p_max=p_max,
            model=model,
            Nterms=Nterms,
            **kwargs)

        period_grid = np.linspace(p_min,p_max,grid)
        spectrum = np.zeros(grid)

        for period,error,amp in zip(periods,period_errors,amplitudes):
            if error < (p_max-p_min)/grid:
                spectrum[np.argmin(abs(period_grid-period))]=amp
            else:
                spectrum += amp*np.exp(-(period_grid-period)**2/(2*error**2))
        return period_grid, spectrum

    def classify(self):
        '''
        performs the classification of this object based on provided photometric data.
        TODO: this is going to be a big function and requires a lot of work!
        '''
        # self.type = 'RRab'
        raise NotImplementedError

    def open_widget(self):
        raise NotImplementedError('in development')

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

    def calc_FAP(self,period,):
        '''
        ** THIS FUNCTION IS WORK IN PROGRESS **
        Calculates the false alarm probability of signal at specified period.
        Based on VanderPlas+ 2018.
        inputs:
            period
            method: {'range','freq','Baluev'}
        '''
        raise NotImplementedError
 
    def get_epoch_offset(self,period=None,x=None,y=None,yerr=None,model='Fourier',N=1000,Nterms=5,**kwargs):
        '''
        TODO: define the 'maxima': is it the minimum in magnitude or maximum in any value? current implementation -> 'magnitude' interpretation only
        inputs:
            N: number of samples across the phase (single period). The peak should have width W >> P/1000.
        '''
        # use default values if data is not explicitly given
        x,y,yerr = self.prepare_data(x,y,yerr)

        # use automatically determined period if period is not explicitly given
        if period == None:
            if self.period == None:
                period, _ = self.get_period(**kwargs)
            period = self.period
        
        # get the phase offset (phase of maxima for raw data)
        x_th = np.linspace(0,period,N)
        _, y_th = self.get_bestfit_curve(x=x,y=y,yerr=yerr,period=period,model=model,Nterms=Nterms,x_th=x_th)
        epoch_offset = x_th[np.argmin(y_th)]
        self.epoch_offset = epoch_offset
        return epoch_offset
