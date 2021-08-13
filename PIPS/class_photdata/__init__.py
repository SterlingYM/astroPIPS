import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import numba
from multiprocessing import Pool
import time
import warnings
warnings.simplefilter("ignore", OptimizeWarning)
import copy

from ..periodogram import Periodogram
from ..periodogram.custom import periodogram_custom, get_bestfit, check_MODEL_KWARGS, MODELS, P0_FUNCS
from ..periodogram.custom import get_chi2 as _get_chi2
from ..periodogram.linalg import periodogram_fast
from ..periodogram.custom.models.Fourier import fourier, fourier_p0
from ..periodogram.custom.models.Gaussian import gaussian, gaussian_p0

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
        get_period(self,p_min=0.1,p_max=4,x=None,y=None,yerr=None,Nterms=5,method='fast',model='Fourier',peaks_to_test=5,R_peak=500,debug=False,force_refine=False,default_err=1e-6,**kwargs)
        get_period_multi(self,N,FAR_max=1e-3,model='Fourier',Nterms=5,**kwargs)
        amplitude_spectrum(self,p_min,p_max,N,model='Fourier',grid=10000,plot=False,Nterms=5,**kwargs)
        get_bestfit(N,model='Fourier',period=None,plot=True,return_curve=False,return_params=False)
        classify(self)
        open_widget(self)
        plot_lc(self,period=None,invert_yaxis=True,**kwargs)
    '''
    
    def __init__(self,data,label='',band=None):
        '''
        Takes in a list or numpy array of time-series data
        e.g. ```[time,mag,mag_err]```
        '''
        self.x = np.array(data[0])
        self.y = np.array(data[1])
        self.yerr = np.array(data[2])
        self.period = None
        self.period_err = None
        self.amplitude = None
        self.amplitude_err = None
        self.label = label
        self.band = band
        self.epoch = None
        self.epoch_offset = None
        self.meanmag = None # based on best-fit function: requires period
        self.multiprocessing = True
        self.periodogram = Periodogram(photdata=self)

    def __repr__(self):
        return f"Photdata ({self.label},{self.band},{len(self.x)},{self.period})"

    def __str__(self):
        return f"Photdata {self.label}: band={self.band}, size={len(self.x)}, period={self.period}"

    def __len__(self):
        return len(self.x)

    def __hash__(self):
        if hasattr(self,'_x_raw'):
            return hash((self._x_raw.tobytes(), self._y_raw.tobytes(), self._yerr_raw.tobytes()))
        return hash((self.x.tobytes(), self.y.tobytes(), self.yerr.tobytes()))

    def __eq__(self,other):
        return hash(self) == hash(other)

    def __ne__(self,other):
        return hash(self) != hash(other)

    def __add__(self,other):
        _x = [*self.x,*other.x]
        _y = [*self.y,*other.y]
        _yerr = [*self.yerr,*other.yerr]
        return type(self)([_x,_y,_yerr])

    def __copy__(self):
        '''deep copy by default!'''
        newone = type(self)(copy.deepcopy(self.data))
        newone.__dict__.update(copy.deepcopy(self.__dict__))
        return newone

    def copy(self):
        return self.__copy__()

    @property
    def shape(self):
        return np.array([self.x,self.y,self.yerr]).shape

    @property
    def data(self):
        return np.array([self.x,self.y,self.yerr])
        
    ##############
    # utilities
    ##############
    def check_model(self, model, p0_func, **kwargs):
        """
        Checks that a given input model is available.
        model : (str/obj) user-input model.
        p0_func : (str/obj) dictionary containing model strings as keys and arbitrary functions as values.
        """
        if isinstance(model, str):
            MODEL = MODELS[model]
            KWARGS = check_MODEL_KWARGS(model,**kwargs)
            P0_FUNC = P0_FUNCS[model]
        elif hasattr(model, '__call__'):
            MODEL = model
            KWARGS= check_MODEL_KWARGS(model,**kwargs)
            if hasattr(p0_func, '__call__'):
                P0_FUNC = p0_func
            else:
                raise ValueError('custom model requires initial-guess prep function (p0_func).')
        else:
            raise ValueError('model has to be either a function or a pre-defined function name')
        return MODEL, P0_FUNC, KWARGS
        
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
            self._x_raw = self.x
            self._y_raw = self.y
            self._yerr_raw = self.yerr

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
        condition = np.full(self._x_raw.shape, True, dtype=bool)
        if self.cut_xmin is not None:
            condition = condition & (self._x_raw >= self.cut_xmin)
        if self.cut_xmax is not None:
            condition = condition & (self._x_raw <= self.cut_xmax)
        if self.cut_ymin is not None:
            condition = condition & (self._y_raw >= self.cut_ymin)
        if self.cut_ymax is not None:
            condition = condition & (self._y_raw <= self.cut_ymax)
        if self.cut_yerr_min is not None:
            condition = condition & (self._yerr_raw >= self.cut_yerr_min)
        if self.cut_yerr_max is not None:
            condition = condition & (self._yerr_raw <= self.cut_yerr_max)

        # apply cuts
        self.x = self._x_raw[condition]
        self.y = self._y_raw[condition]
        self.yerr = self._yerr_raw[condition]
        
    def reset_cuts(self):
        '''
        resets cuts applied by cut() function.
        '''
        if hasattr(self,'_x_raw'):
            self.cut_xmin = None
            self.cut_xmax = None
            self.cut_ymin = None
            self.cut_ymax = None
            self.cut_yerr_min = None
            self.cut_yerr_max = None
            self.x = self._x_raw
            self.y = self._y_raw
            self.yerr = self._yerr_raw

    def summary(self):
        '''
        prints out the summary.
        TODO: Jupyter widget?
        '''
        return self.__str__()

    def prepare_data(self,x,y,yerr):
        if (x is None) and (y is None) and (yerr is None):
            x = self.x
            y = self.y
            yerr = self.yerr
        elif not ((x is not None) and (y is not None) and (yerr is not None)):
            raise ValueError('Input data is incomplete. All x, y, and yerr are needed.')
        return x,y,yerr

    def get_bestfit_curve(self,x=None,y=None,yerr=None,period=None,model='Fourier',p0_func=None,x_th=None,return_params=False,return_param_err=False,use_original_x=False,**kwargs):
        '''
        Calculates the best-fit smooth curve.
        '''
        # prepare data
        if model=='Fourier':
            if 'Nterms' in kwargs:
                Nterms = kwargs['Nterms']
            else:
                kwargs['Nterms'] = 5        
        x,y,yerr = self.prepare_data(x,y,yerr)
        
        # use automatically determined period if period is not explicitly given
        if period == None:
            if self.period == None:
                period, _ = self.get_period(model=model,p0_func=p0_func,**kwargs)
            period = self.period

        # select models
        MODEL, P0_FUNC, KWARGS = self.check_model(model, p0_func, kwarg_for_helper=True,**kwargs)

        # get bestfit model-parameters
        popt,pcov = get_bestfit(MODEL,P0_FUNC,x,y,yerr,period,return_yfit=False,return_params=True,return_pcov = True,**KWARGS)
        if return_params:
            if return_param_err:
                return popt,np.sqrt(np.diag(pcov))
            return popt

        # construct theoretical curve
        MODEL, P0_FUNC, KWARGS = self.check_model(model, p0_func, kwarg_for_helper=False,**kwargs)
        if x_th is None:
            if use_original_x:
                x_th = self.x
            else:
                x_th = np.linspace(0,period,1000)
        y_th = MODEL(x_th,period,np.array(popt),**KWARGS)

        return x_th,y_th

    def get_chi2(self,x=None,y=None,yerr=None,period=None,model='Fourier',p0_func=None,x_th=None,**kwargs):
        '''
        Calculates the best-fit smooth curve.
        '''
        # prepare data
        if model=='Fourier':
            if 'Nterms' in kwargs:
                Nterms = kwargs['Nterms']
            else:
                kwargs['Nterms'] = 5        
        x,y,yerr = self.prepare_data(x,y,yerr)
        
        # use automatically determined period if period is not explicitly given
        if period == None:
            if self.period == None:
                period, _ = self.get_period(model=model,p0_func=p0_func,**kwargs)
            period = self.period

        # select models
        MODEL, P0_FUNC, KWARGS = self.check_model(model, p0_func, kwarg_for_helper=True,**kwargs)

        # get bestfit chi-square
        chi2 = _get_chi2(MODEL,P0_FUNC,x,y,yerr,period,**KWARGS)
        return chi2

    def get_bestfit_amplitude(self,x=None,y=None,yerr=None,period=None,model='Fourier',Nterms=5,**kwargs):
        '''
        calculates the amplitude of best-fit curve.
        '''
        _,y_th = self.get_bestfit_curve(x,y,yerr,period,model,Nterms,**kwargs)
        return np.max(y_th)-np.min(y_th)

    def get_meanmag(self,x=None,y=None,yerr=None,period=None,model='Fourier',Nterms=5,**kwargs):
        '''
        calculates an estimated mean magnitude from best-fit curve.
        This method requires a reliable fitting, but is more robust against incomplete sampling in pulsation phase
        '''
        _,y_th = self.get_bestfit_curve(x,y,yerr,period,model,Nterms,**kwargs)
        return np.mean(y_th)

    def get_SR(self,power):
        return power / power.max()

    def get_SDE(self,power,peak_only=False):
        SR = self.get_SR(power)
        if peak_only:
            return (1-SR.mean())/SR.std()
        else:
            return (SR-SR.mean())/SR.std()

    def plot_lc(self,period=None,invert_yaxis=True,figsize=(8,4),ax=None,return_axis=False,title=None,plot_bestfit=False,model_color='yellowgreen',model_kwargs={},ylabel='mag',**kwargs):
        '''
        plots phase-folded light curve.
        '''
        if period is None:
            if self.period is None:
                raise ValueError('folding period needs to be specified')
            else:
                period = self.period
        phase = (self.x % period)/period

        if title is None:
            title = self.label
        
        # plot
        if ax==None:
            fig, ax = plt.subplots(1,1,figsize=figsize)
        if 'color' not in kwargs.keys():
            kwargs['color'] = 'k'
        if 'fmt' not in kwargs.keys():
            kwargs['fmt'] = 'o'
        if 'ms' not in kwargs.keys():
            kwargs['ms'] = 2
        ax.errorbar(phase,self.y,self.yerr,**kwargs)
        ax.errorbar(phase+1,self.y,self.yerr,**kwargs)
        ax.set_title(title,fontsize=16)
        ax.set_xlabel('Phase',fontsize=16)
        ax.set_ylabel(ylabel,fontsize=16)

        # options
        if invert_yaxis and not ax.yaxis_inverted():
            ax.invert_yaxis()
        if plot_bestfit:
            x_th,y_th = self.get_bestfit_curve(**model_kwargs)
            plt.plot(x_th/period,y_th,lw=3,c=model_color)
            plt.plot(x_th/period+1,y_th,lw=3,c=model_color)
        if return_axis:
            return ax
 
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

    #################
    # analysis tools
    #################
     
    def get_period(self,repr_mode='likelihood',return_Z=False,**kwargs):
        if repr_mode=='chisq' or repr_mode=='chi2' or repr_mode=='chi_square':
            return self._get_period(**kwargs)
        if repr_mode in ['likelihood','lik','log-likelihood','loglik']:
            period,period_err,Z = self._get_period_likelihood(repr_mode=repr_mode,**kwargs)
            if return_Z:
                return period,period_err,Z
            else:
                return period,period_err

    def _get_period(self,p_min=0.1,p_max=4,x=None,y=None,yerr=None,
        method='fast',model='Fourier',p0_func=None,
        peaks_to_test=5,R_peak=500,N0=10,debug=False,force_refine=False,
        default_err=1e-6,no_overwrite=False,multiprocessing=True,
        return_SDE=False,ignore_warning=False,
        try_likelihood=False,**kwargs):
        '''
        detects period.
        '''
        # check global setting for mp
        multiprocessing = multiprocessing and self.multiprocessing

        # model & kwargs preparation     
        if method=='fast':
            if 'Nterms' in kwargs:
                Nterms = kwargs['Nterms']
            else:
                kwargs['Nterms'] = 5
        if model=='Fourier':
            Nterms = kwargs['Nterms']   
        MODEL, P0_FUNC, KWARGS = self.check_model(model,p0_func,**kwargs)


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
        period,power = self.periodogram(p_min=p_min,p_max=p_max,x=x,y=y,yerr=yerr,
                            method=method,model=model,p0_func=p0_func,N0=N0,
                            multiprocessing=multiprocessing,**kwargs)

        # calculate peak SDE
        period_SDE = self.get_SDE(power,peak_only=True)

        # select top peaks_to_test independent peaks
        if debug:
            print(f'{time.time()-t0:.3f}s --- detecting top {peaks_to_test} peaks...')
        peak_idx = []

        T = x.max()-x.min()
        peak_width = p_min**2 *T / (T**2-0.25)
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
            custom_periods = np.concatenate((custom_periods,np.linspace(peak-peak_width,peak+peak_width,R_peak)))
        if debug:
            print(f'{time.time()-t0:.3f}s --- performing finer sampling near peaks...')
        period,power = self.periodogram(
            custom_periods=custom_periods,N0=N0,
            x=x,y=y,yerr=yerr,method=method,model=model,p0_func=p0_func,
            multiprocessing=multiprocessing,**kwargs
            )
        period = period[power==power.max()][0]
        if debug:
            print(f'{time.time()-t0:.3f}s --- period candidate: ', period)
            
        model_bestfit = get_bestfit(MODEL,P0_FUNC,x,y,yerr,period,return_yfit=False,return_params=True,**kwargs)

        # detect aliasing
        if model=='Fourier':
            if debug: 
                print(f'{time.time()-t0:.3f}s --- detecting aliasing...')
            factor = np.argmax(abs(model_bestfit[1:Nterms]))+1
            if factor != 1:
                period /= factor
                model_bestfit = get_bestfit(MODEL,P0_FUNC,x,y,yerr,period,return_yfit=False,return_params=True,**KWARGS)
            if debug:
                print(f'{time.time()-t0:.3f}s --- alias factor: ',factor)
                print(f'{time.time()-t0:.3f}s --- period candidate: ',period)

        # get uncertainty
        if debug:
            print(f'{time.time()-t0:.3f}s --- estimating the uncertainty...')
        
        KWARGS['maxfev'] = 100000
        popt, pcov = get_bestfit(MODEL,P0_FUNC,x,y,yerr,period,return_yfit=False,return_params=True,return_pcov=True,fit_period=True,**KWARGS)
        period_err = np.sqrt(np.diag(pcov))[0]
        if debug: 
            print(f'{time.time()-t0:.3f}s --- period candidate: ',period)
            print(f'{time.time()-t0:.3f}s --- period fitted*: ',popt[0])
            print(f'{time.time()-t0:.3f}s --- period error: ',period_err)
        if period_err == np.inf:
            # automatically activate the refinement process
            period_err = 0 

        # re-sample if sampling size is not fine enough
        if (period_err < (2*peak_width/R_peak)*10) or force_refine:
            if debug:
                print(f'{time.time()-t0:.3f}s --- refining samples...')
                print(f'{time.time()-t0:.3f}s --- refining search width = {peak_width/10:.3e}')

            # prepare new search width -- narrower and thus refined
            #TODO: discuss this method
            new_search_width = peak_width/R_peak*100 
            custom_periods = np.linspace(period-new_search_width,period+new_search_width,R_peak)

            # get periodogram
            periods,power = self.periodogram(
                custom_periods=custom_periods,N0=N0,
                x=x,y=y,yerr=yerr,method=method,model=model,p0_func=p0_func,multiprocessing=multiprocessing,**kwargs
                )
            period = periods[power==power.max()][0]

            # get uncertainty
            KWARGS['maxfev'] = 100000
            popt, pcov = get_bestfit(MODEL,P0_FUNC,x,y,yerr,period,return_yfit=False,return_params=True,return_pcov=True,fit_period=True,**KWARGS)
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
            if not ignore_warning:
                warningMessage = 'warning: provided uncertainty may not be accurate. Try increasing sampling size (N0, default 10).'
                print(warningMessage)
        elif debug:
            print(f'{time.time()-t0:.3f}s --- * period error validated')

        if period_err == np.inf:
            print('warning: error size infinity: replacing with periodogram peak width')
            period_err = peak_width

        # finalize
        if not no_overwrite:
            self.period = period
            self.period_err = period_err
            self.period_SDE = period_SDE
        if debug:
            print(f'{time.time()-t0:.3f}s ---','period = {:.{precision}f} +- {:.{precision}f}d'.format(period,period_err,precision=5 if period_err==np.inf else int(abs(np.log10(period_err))+2)))
            print(f'{time.time()-t0:.3f}s --- process completed.')

        if return_SDE == True:
            return period,period_err,period_SDE
        return period,period_err
   
    def _get_period_likelihood(self,period=None,period_err=None,p_min=0.1,p_max=4.0,N_peak=1000,N_noise=5000,Nsigma_range=10,return_SDE=False,repr_mode='likelihood',**kwargs):
        '''
        Calculates the period, uncertainty, and significance based on the given initial guesses.
        '''
        if period is None and period_err is None:
            if return_SDE:
                period, period_err,SDE = self._get_period(p_min=p_min,p_max=p_max,return_SDE=return_SDE,**kwargs)
            else:
                period, period_err = self._get_period(p_min=p_min,p_max=p_max,**kwargs)

        def log_Gaussian(x,mu,sigma,offset):
            # return amp*np.exp(-0.5*(x-mu)**2/sigma**2)
            return -0.5*(x-mu)**2/sigma**2 + offset

        # sample likelihood near the period
        periods,lik = self.periodogram(
            p_min = period-period_err*Nsigma_range,
            p_max = period+period_err*Nsigma_range,
            N=N_peak,
            repr_mode='loglik',
            raise_warnings=False,
            **kwargs
            )
        popt,_ = curve_fit(log_Gaussian,periods,lik,p0=[period,period_err,lik.max()],bounds=[[0,0,-np.inf],[np.inf,np.inf,np.inf]])
        signal_log = lik.max()
        period_mu,period_sigma,_ = popt

        # sample likelihood for shuffled data
        idx = np.arange(len(self.x))
        np.random.shuffle(idx)
        y_noise = self.y[idx]
        yerr_noise = self.yerr[idx]
        _,loglik_noise = self.periodogram(
            p_min=p_min, p_max=p_max, N=N_noise,
            x=self.x, y=y_noise, yerr=yerr_noise,
            repr_mode = 'log-likelihood',
            raise_warnings=False,
            **kwargs
            )
        noise_mu,noise_sigma = loglik_noise.mean(),loglik_noise.std()
        Zscore = (signal_log-noise_mu)/noise_sigma
        
        if return_SDE:
            return period_mu,period_sigma,Zscore, SDE
        return period_mu, period_sigma, Zscore
        

    def get_period_multi(self,N,FAR_max=1e-3,model='Fourier',p0_func=None,**kwargs):
        '''
        multi-period detection. 
        Re-detects P1 and then proceeds to P2, P3, ... PN.
        Pn=None if FAR for nth period exceeds given thershold.
        '''
        # TODO: implement FAR

        # model & kwargs preparation     
        if model=='Fourier':
            if 'Nterms' in kwargs:
                Nterms = kwargs['Nterms']
            else:
                kwargs['Nterms'] = 5
            Nterms = kwargs['Nterms']   
        MODEL, P0_FUNC, KWARGS = self.check_model(model,p0_func,**kwargs)
        
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
                model=model,
                p0_func=p0_func,
                **kwargs)
            periods.append(period)
            period_errors.append(period_err)
            amp = self.get_bestfit_amplitude(
                x=x_prewhitened,
                y=y_prewhitened,
                yerr=yerr_prewhitened,
                period=period,
                model=model,
                **kwargs)
            amplitudes.append(amp)
            y_prewhitened -= get_bestfit(
                MODEL,P0_FUNC,
                x_prewhitened,
                y_prewhitened,
                yerr_prewhitened,
                period,
                return_yfit=True,return_params=False,**KWARGS)
        return periods,period_errors,amplitudes

    def amplitude_spectrum(self,p_min,p_max,N,model='Fourier',p0_func=None,grid=10000,plot=False,**kwargs):
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
            p0_func=p0_func,
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

