from . import *
from .linalg import periodogram_fast
from .custom import periodogram_custom
import matplotlib.pyplot as plt
import numpy as np

class Periodogram:
    def __init__(self,photdata=None,periods=None,power=None,kwargs=None,mode='regular'):
        self.photdata = photdata
        self.periods = periods
        self.power = power
        self.kwargs = kwargs # will be filled once __call__ is run
        self.mode = mode
        self.current = 0

    def __call__(self,**kwargs):
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
            method='fast',x=None,y=None,yerr=None,plot=False,
            multiprocessing=True,N0=5,model='Fourier',
            raise_warnings=True,**kwargs):
        '''
        Generate periodogram. 
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
                'p_min':p_min,'p_max':p_max,'custom_periods':custom_periods,'N':N,'x':x,'y':y,'yerr':yerr,'multiprocessing':multiprocessing,'model':model
                },
            'custom':{
                'p_min':p_min,'p_max':p_max,'custom_periods':custom_periods,'N':N,'x':x,'y':y,'yerr':yerr,'multiprocessing':multiprocessing,'model':model
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
        if return_axis:
            return ax

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

    def SR(self):
        SR = self.power / self.power.max()
        return Periodogram(self.photdata,self.periods,SR,self.kwargs,mode='SR')

    def SDE(self,peak_only=False):
        periods, SR = self.SR()
        if peak_only:
            return  (1-SR.mean())/SR.std()
        else:
            SDE = (SR-SR.mean())/SR.std()
            return Periodogram(self.photdata,self.periods,SDE,self.kwargs,mode='SDE')