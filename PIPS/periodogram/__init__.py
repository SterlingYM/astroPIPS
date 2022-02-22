from PIPS.periodogram import *
from PIPS.periodogram.linalg import periodogram_fast
from PIPS.periodogram.custom import periodogram_custom
import matplotlib.pyplot as plt
import numpy as np

class Periodogram:
    """ An object that generates and plots the periodogram.
    
    Attributes:
        photdata (PIPS.photdata): the photdata object that contains the photometric data to construct periodograms.
        periods (numpy array): a list of period values at which the periodogram is evaluated.
        power (numpy array): a list of periodogram values.
        kwargs (dict): arguments to be passed to specific methods of periodograms.
        mode (str): the periodogram mode. Internal flagging purpose only.
    """
    
    def __init__(self,photdata=None,periods=None,power=None,kwargs=None,mode='regular'):
        """ Initializes the periodogram object.
        
        Args:
            photdata (PIPS.photdata): the photdata object that contains the photometric data to construct periodograms.
            periods (numpy array): a list of period values at which the periodogram is evaluated.
            power (numpy array): a list of periodogram values.
            kwargs (dict): arguments to be passed to specific methods of periodograms.
            mode (str): the periodogram mode. Internal flagging purpose only.
        """
        self.photdata = photdata
        self.periods = periods
        self.power = power
        self.kwargs = kwargs # will be filled once __call__ is run
        self.mode = mode
        self.current = 0

    def __call__(self,**kwargs):
        """ Evaluates the periodogram.
        
        Returns:
            self (PIPS.periodogram): the periodogram object with updated periodogram values.
        """
        periods,power = self._periodogram(**kwargs)
        self.periods = periods
        self.power = power
        self.kwargs = kwargs
        return self

    @property
    def data(self):
        return [self.periods,self.power]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < 2:
            self.current += 1
            return self.data[self.current-1]
        else:
            self.current = 0
            raise StopIteration

    def _periodogram(self,p_min=0.1,p_max=4,custom_periods=None,N=None,
            method='fast',x=None,y=None,yerr=None,
            multiprocessing=False,N0=10,model='Fourier',
            raise_warnings=True,**kwargs):
        ''' Generates periodogram. 
        
        Args:
            p_min (float): the minimum value of the period to evaluate.
            p_max (float): the maximum value of the period to evaluate.
            custom_periods (numpy array): a list of specific values of period at which the periodogram is evaluated. Using this overrides the default setting of equally-spaced grid of period values between p_min and p_max.
            N (int): the number of samples for equally-spaced period grid between p_min and p_max.
            method (str)['fast','custom']: switches between linear algebra (fast) and linear regression (custom) methods. Fast mode is only available for the Fourier model.
            x: the time data.
            y: the mag/flux data.
            yerr: the uncertainties in mag/flux data.
            multiprocessing: the option to use multiprocessing feature. False by default.
            N0: the ratio between the grid size and the expected period width. See our paper for detail.
            model (str/obj): the light curve model. It has to be the name of the pre-implemented methods or a user-implemented function in a specific format. See tutorial for detail.
            raise_warnings (bool): setting this to True prints out a warning message when provided N is smaller than the recommended size.
            
        Returns:
            periods: the period values at which the periodogram is evaluated.
            power: the values of the periodogram.
        '''
        # call parent class data
        photdata = self.photdata

        # check global setting for mp
        multiprocessing = multiprocessing and self.photdata.multiprocessing

        # prepare data
        x,y,yerr = photdata.prepare_data(x,y,yerr)
        
        # prepare period-axis
        if isinstance(custom_periods,(list,np.ndarray, np.generic)):
            # if custom_periods is given: just use them
            pass
        else:
            # determine sampling size
            T = x.max()-x.min()
            N_auto = int((1/p_min-1/p_max)/(1/T)*N0)
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

        # default Nterms handling: TODO: make this more elegant
        if method=='fast' or model=='Fourier':
            if 'Nterms' in kwargs:
                Nterms = kwargs['Nterms']
            else:
                kwargs['Nterms'] = 5
                Nterms = 5

        METHOD_KWARGS = {
            'fast': {
                'p_min':p_min,
                'p_max':p_max,
                'custom_periods':custom_periods,
                'N':N,
                'x':x,
                'y':y,
                'yerr':yerr,
                'multiprocessing':multiprocessing,
                'model':model
                },
            'custom':{
                'p_min':p_min,
                'p_max':p_max,
                'custom_periods':custom_periods,
                'N':N,
                'x':x,
                'y':y,
                'yerr':yerr,
                'multiprocessing':multiprocessing,
                'model':model
                }
        }
        kwargs.update(METHOD_KWARGS[method])
        
        # main
        periods,power = METHODS[method](**kwargs)
        
        if 'repr_mode' in kwargs:
            if kwargs['repr_mode'] in ['likelihood','lik','log-likelihood','loglik']:
                self.mode='likelihood'
        periods = np.asarray(periods)
        power   = np.asarray(power)
        return periods,power

    def zoom(self,p_min=None,p_max=None,width_factor=2,**kwargs):
        """ zooms into the peak.
        
        Args:
            p_min: the minimum period of the new range.
            p_max: the maximum period of the zoom range.
            width_factor: the ratio between the width of the new range around the peak and the analytic, expected width of the peak. This is only used when p_min and p_max are not provided.
            
        Returns:
            periodogram (PIPS.periodogram): the periodogram object with the new range of periodogram.
        """
        kwargs = kwargs.update({'return_axis':True})
        period_at_max = self.periods[self.power==self.power.max()]

        if p_min==None and p_max==None:
        # estimate peak width
            T = self.photdata.x.max()-self.photdata.x.min()
            peak_width = period_at_max**2 *T / (T**2-0.25)
            width = peak_width * width_factor
            p_min = period_at_max - width
            p_max = period_at_max + width

        cut = (self.periods >= p_min) & (self.periods <= p_max)
        new_periods = self.periods[cut]
        new_power = self.power[cut]
        return Periodogram(self.photdata,new_periods,new_power,self.kwargs,mode=self.mode)

    def refine(self,factor=10):
        """ performs subsampling of periodogram to refine the quality.
        
        Args:
            factor: the ratio between the old grid size and the new grid size.
        
        Returns:
            periodogram (PIPS.periodogram): a periodogram object that contains refined periodogram.
        """
        p_min = self.periods.min()
        p_max = self.periods.max()
        N = len(self.periods)
        N_new = N * factor

        custom_periods = 1/np.linspace(1/p_max,1/p_min,N_new)
        kwargs = {**self.kwargs}
        kwargs.update({
            'p_min':None,
            'p_max':None,
            'custom_periods':custom_periods
            })
        periods,power = self._periodogram(**kwargs)
        return Periodogram(self.photdata,periods,power,self.kwargs,mode=self.mode)

    def plot(self,ax=None,return_axis=False,c='#454545',show_peak=False,**kwargs):
        """ automatically plots the periodogram.
        
        Args:
            ax (matplotlib.axis.Axes): the user-defined matplotlib axis.
            return_axis (bool): the option to return matplotlib axis.
            c (str): the color of periodogram. Follow the matplotlib syntax.
            show_peak (bool): an option to plot a vertical line at the peak.
            
        Returns:
            ax (matplotlib.axis.Axes): the matplotlib axis with the periodogram plotted.
        """
        periods,power = self

        if ax==None:
            fig,ax = plt.subplots(1,1,figsize=(13,3))

        # plot
        ax.plot(periods,power,c=c,lw=1)
        if show_peak:
            ax.axvline(periods[power==power.max()],c='orange',lw=1.5,linestyle=':',zorder=10)
        ax.set_xlabel('period',fontsize=15)
        ax.set_xlim(periods.min(),periods.max())

        peak_period = periods[power==power.max()]


        # switch between different types
        if self.mode == 'regular':
            ax.fill_between(periods,0,power,color=c)
            ax.set_ylim(0,1)
        elif self.mode == 'SR':
            ax.fill_between(periods,0,power,color=c)
            ax.set_ylim(0,1)
        elif self.mode == 'SDE':
            ax.fill_between(periods,0,power,color=c)
            ax.set_ylim(0,power.max())
        elif self.mode == 'likelihood':
            ax.fill_between(periods,power.min(),power,color=c)
            ax.set_ylim(power.min(),power.max()*1.05)

        if return_axis:
            return ax
            
    def SR(self):
        """ Evaluates the SR. See our paper for detail.
        
        Returns:
            Periodogram (PIPS.periodogram): the periodogram that contains SR instead of the raw periodogram.
        """
        SR = self.power / self.power.max()
        return Periodogram(self.photdata,self.periods,SR,self.kwargs,mode='SR')

    def SDE(self,peak_only=False):
        """ Calculates SDE. See our paper for discussion.
        
        Args:
            peak_only (bool): the option to return the peak SDE value only.
        
        Returns:
            SDE (float): the SDE value at the peak. This is returned when peak_only==True.
            periodogram (PIPS.periodogram): the periodogram object that contains SDE instead of the raw periodogram power.
        
        """
        periods, SR = self.SR()
        if peak_only:
            return  (1-SR.mean())/SR.std()
        else:
            SDE = (SR-SR.mean())/SR.std()
            return Periodogram(self.photdata,self.periods,SDE,self.kwargs,mode='SDE')
