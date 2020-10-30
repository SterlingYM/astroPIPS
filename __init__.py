### Period Determination and Identification Pipeline Suite (PIPS) ###
### this is a tool for variable star analysis and is under development.
### Yukei S. Murakami, UC Berkeley 2020
### sterling.astro@berkeley.edu

import warnings
warnings.simplefilter(action='ignore')

import numpy as np
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time

#import seaborn as sns
#sns.set()



def data_readin_LPP(path,filter='V'):
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

class photdata:
    ###########################
    ### Initialization 
    ###########################
    def __init__(self,data):
        # data
        self.data = data
        self.t       = data[0]
        self.mag     = data[1]
        self.mag_err = data[2]
        self.x = self.t
        self.y = self.mag
        self.yerr = self.mag_err
        self.period = None
        self.period_err = None
        self.amplitude = None
        self.p1 = None
        self.p2 = None
        self.label = ''
        self.epoch = None

        # options
        self.A0 = 15 # initial guess for mean mag (zeroth order in Fourier series)
        self.K=5
        self.err_cut_threshold = None
        self.max_mag_threshold = None
        self.min_mag_threshold = None
        self.quiet = True
        self.LS_min_f = 0.5
        self.LS_max_f = 10 
        
        
    ###########################    
    ### Fourier Functions 
    ###########################
    def fourier_composition(self,t,omega,A0,*ab_list):
        k_list = np.array(range(self.K))+1
        a_list = ab_list[:self.K]
        b_list = ab_list[self.K:]    
        return_val = A0
        for k,a,b in zip(k_list,a_list,b_list):
            return_val += a*np.sin(k*omega*t) + b*np.cos(k*omega*t)
        return return_val
        
    def fourier_composition_folded(self,x,period,A0,*ab_list):
        omega = 2*np.pi/period
        t = (np.array(x) % period)
        y_fit = self.fourier_composition(t,omega,A0,*ab_list)
        return y_fit

    def twoperiod_model_nonsinusoidal(self,x,p1,p2,A0,A1,A2,phi1,phi2):
        popt_RRab_template = [-0.59720581,  0.2698969 , -0.20855331,  0.07600764, -0.00697336, \
                              -0.41523584,  0.11322467, -0.13080449,  0.12862137, -0.05014972]
        y = A0
        for i in range(5):
            k  = i+1
            y += A2*(popt_RRab_template[i] * np.sin(k*2*np.pi/p2*x - phi2)+\
                     popt_RRab_template[i+5]*np.cos(k*2*np.pi/p2*x - phi2))
        y += A1*np.sin(2*np.pi/p1*x - phi1)
        return y    

    def twoperiod_model_freeform(self,x,p1,p2,A0,*popts):
        popt1 = popts[:10]
        popt2 = popts[10:]
        y = A0
        for i in range(5):
            k  = i+1
            y += popt1[i] * np.sin(k*2*np.pi/p1*x)+popt1[i+5]*np.cos(k*2*np.pi/p1*x)
            y += popt2[i] * np.sin(k*2*np.pi/p2*x)+popt2[i+5]*np.cos(k*2*np.pi/p2*x)
        return y

    def simple_fourier(self,x,p,A0,*popt):
        y = A0
        for i in range(5):
            k = i+1
            y += popt[i]*np.sin(k*2*np.pi/p*x) + popt[i+5]*np.cos(k*2*np.pi/p*x)
        return y

    ###########################    
    ### Data Manipulation Functions
    ###########################    
    def phot_err_cut(self,Nsigma=1):
        # cuts data with uncertainties larger than threshold
        if self.err_cut_threshold == None:
            self.err_cut_threshold = np.mean(self.yerr)+Nsigma*np.std(self.yerr)
        x1 = self.x[self.yerr<self.err_cut_threshold]
        y1 = self.y[self.yerr<self.err_cut_threshold]
        yerr1 = self.yerr[self.yerr<self.err_cut_threshold]
        return photdata([x1,y1,yerr1])
   
    def phot_mag_cut(self,min_mag=None,max_mag=None):
        # cuts data with magnitude outside of the threshold
        if min_mag != None:
            self.min_mag_threshold = min_mag
        if max_mag != None:
            self.max_mag_threshold = max_mag
        if self.min_mag_threshold != None:
            x1 = self.x[self.y>self.min_mag_threshold]
            y1 = self.y[self.y>self.min_mag_threshold]
            yerr1 = self.yerr[self.y>self.min_mag_threshold]
        else:
            x1,y1,yerr1 = self.x, self.y, self.yerr
        if self.max_mag_threshold != None:
            x2 = x1[y1<self.max_mag_threshold]
            y2 = y1[y1<self.max_mag_threshold]
            yerr2 = yerr1[y1<self.max_mag_threshold]
        else:
            x2,y2,yerr2 = x1,y1,yerr1
        return photdata([x2,y2,yerr2])

            
    
    ###########################    
    ### Data Analysis Functions
    ###########################    
    def calc_chisq(self,FF_popt):
        # requires a set of optimized parameters for fourier fitting self.FF_popt
        period = FF_popt[0]
        y_th  = self.fourier_composition(self.x,2*np.pi/period,*FF_popt[1:])
        chisq = np.sum((self.y-y_th)**2/self.yerr)
        return chisq,len(self.x)
    
    def get_best_fit_at_p(self,period):
        ab = np.full(2*self.K,1)
        p0=[self.A0,*ab]
        x_folded = self.x%period
        omega = 2*np.pi/period
        popt,_ = curve_fit(
                lambda t,*params:self.fourier_composition(t,omega,*params),
                x_folded,self.y,p0=p0,sigma=self.yerr,maxfev=10000)  
        return popt
        
    def test_global_potential_engine(self,p_test):
        # a helper function for multiprocessing in self.test_global_potential()
        ab = np.full(2*self.K,1)
        p0=[p_test,self.A0,*ab]
        FF_popt,FF_pcov = curve_fit(self.fourier_composition_folded,self.x,self.y,p0=p0,sigma=self.yerr,maxfev=1000000)
        chisq, num_data = self.calc_chisq(FF_popt)
        return [FF_popt[0],chisq/num_data]
        
    def test_global_potential(self,test_p_min,test_p_max,test_num):
        if not self.quiet:
            print('testing predictions on global potential...  ',end='')
        test_p_list = np.linspace(test_p_min,test_p_max,test_num)
        pool = Pool()
        outputs = pool.map(self.test_global_potential_engine,test_p_list)
        pool.close()
        pool.join()
        p_list,chisq_list = np.array(outputs).T
        if not self.quiet:
            print('Done')
        return test_p_list, p_list, chisq_list
    
    def get_global_potential_engine(self,p_test):
        # a helper function for multiprocessing in self.get_global_potential()
        popt = self.get_best_fit_at_p(p_test)
        chisq, num_data = self.calc_chisq([p_test,*popt])
        return chisq/num_data

    def get_global_potential(self,test_p_list):#test_p_min,test_p_max,test_num):
        # test_p_list = np.linspace(test_p_min,test_p_max,test_num)
        if not self.quiet:
            print('computing chi2 potential...  ',end='')
        pool = Pool()
        chisq_list = pool.map(self.get_global_potential_engine,test_p_list)
        pool.close()
        pool.join()
        if not self.quiet:
            print('Done')
        return chisq_list

    def get_LS(self):
        freq,power=LombScargle(self.x,self.y).autopower(minimum_frequency=self.LS_min_f,maximum_frequency=self.LS_max_f)
        p_sorted = np.sort(power)
        f0 = freq[power==p_sorted[-1]]
        f1 = freq[power==p_sorted[-2]]
        f2 = freq[power==p_sorted[-3]]
        f3 = freq[power==p_sorted[-4]]    
        return freq,power,[f0,f1,f2,f3]        

    def fit_doublemode_initial(self,period_manual=None,p_guess=None):
        # biased fitting (sinusoidal + template)
        period   = self.period
        if p_guess==None and period<0.5:
            p_guess = 1/0.74*period
        elif p_guess==None and period>0.5:
            p_guess = 0.74*period
        p0 = [p_guess,15,1,1,0,0]
        
        if period>0.5: #rrab
            popt,pcov = curve_fit(lambda x,*params: self.twoperiod_model_nonsinusoidal(x,params[0],period,*params[1:]),
                                  self.x,self.y,p0,maxfev=10000)        
        else:
            popt,pcov = curve_fit(lambda x,*params: self.twoperiod_model_nonsinusoidal(x,period,*params),
                                  self.x,self.y,p0,maxfev=10000)
        return popt,period

    def get_epoch(self):
        '''calculates epoch from period and lc data'''
        # generates best-fit lc
        popt = self.get_best_fit_at_p(self.period)
        x_th = np.linspace(0,self.period,1000)
        y_th = self.fourier_composition(x_th,2*np.pi/self.period,*popt)
        
        # calculate epoch
        max_th = x_th[y_th==y_th.min()]
        max_data = self.t[self.mag == self.mag.min()]
        offset = max_th - max_data%self.period
        self.epoch = max_data + offset
        return epoch
        
    ###########################    
    ### plotting
    ###########################           
    def potential_plot(self,test_p_list,p_list,chisq_list,chi2_potential):
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(15,8))

        # left top: chi2 (potential) vs. fixed period
        ax1.scatter(test_p_list,chi2_potential,s=4,c='k')
        ax1.set_ylabel('chi2 potential',fontsize=15)
        ax1.set_title('Global chi2 potential',fontsize=15)

        # right top: fixed-period chi2 vs. best-fit chi2
        ax2.scatter(chisq_list,chi2_potential,s=4,c='k')
        ax2.set_title('Chi2 potential vs. chi2 for best-fit',fontsize=15)

        # left bottom: fixed-period (initial guess) vs. best-fit period (single-shot estimation)
        ax3.scatter(p_list,test_p_list,s=4,c='k')
        ax3.set_ylabel('initial guess',fontsize=15)
        ax3.set_xlabel('Estimated period',fontsize=15)
        ax3.set_title('Initial guess period vs. best-fit period',fontsize=15)

        # right bottom: best-fit chi2 vs. fixed-period (initial guess)
        ax4.scatter(chisq_list,test_p_list,s=4,c='k')
        ax4.set_xlabel('chi2 for estimated period',fontsize=15)
        ax4.set_title('Initial guess period vs. chi2 for best-fit',fontsize=15)

        # manually align scaling
        ax13_xscale = max(np.max(test_p_list),np.max(p_list))-min(np.min(test_p_list),np.min(p_list))
        ax24_xscale = np.max(chisq_list)-np.min(chisq_list)
        ax12_yscale = np.max(chi2_potential)-np.min(chi2_potential)
        ax34_yscale = np.max(test_p_list)-np.min(test_p_list)
        ax13_xlim  = [min(np.min(test_p_list),np.min(p_list))-0.1*ax13_xscale,max(np.max(test_p_list),np.max(p_list))+0.1*ax13_xscale]
        ax24_xlim  = [np.min(chisq_list)    -0.1*ax24_xscale, np.max(chisq_list)    +0.1*ax24_xscale]
        ax12_ylim  = [np.min(chi2_potential)-0.1*ax12_yscale, np.max(chi2_potential)+0.1*ax12_yscale]
        ax34_ylim  = [np.min(test_p_list)   -0.1*ax34_yscale, np.max(test_p_list)   +0.1*ax34_yscale]
        ax1.set_xlim(ax13_xlim)
        ax1.set_ylim(ax12_ylim)
        ax2.set_xlim(ax24_xlim)
        ax2.set_ylim(ax12_ylim)
        ax3.set_xlim(ax13_xlim)
        ax3.set_ylim(ax34_ylim)
        ax4.set_xlim(ax24_xlim)
        ax4.set_ylim(ax34_ylim)

        plt.tight_layout()
        plt.show()    
    
    def plot_lc_folded(self,period,title='',fitting=True,invert_y=True):
        plt.figure(figsize=(8,4))
        plt.scatter(self.x%period,self.y,color='k',s=8)
        plt.scatter(self.x%period+period,self.y,color='k',s=8)
        if fitting:
            popt = self.get_best_fit_at_p(period)
            x_th = np.linspace(0,2*period,1000)
            y_th = self.fourier_composition(x_th,2*np.pi/period,*popt)
            plt.plot(x_th,y_th)
        plt.xlabel('time [day]',fontsize=15)
        plt.ylabel('mag',fontsize=15)
        plt.title(title + '  P={:.9f}'.format(period))
        ax = plt.gca()
        if invert_y:
            ax.invert_yaxis()  

    def plot_lc_doublemode(self,p1=None,p2=None,figsize=(20,4),show=True):
        if not (p1 != None and p2 != None):
            p1 = self.p1
            p2 = self.p2
        x,y,yerr = self.x, self.y, self.yerr
        p0 = [15,*np.zeros(20)]
        popt,pcov = curve_fit(lambda x,*params: self.twoperiod_model_freeform(x,p1,p2,*params),
                              x,y,p0,maxfev=10000)
        A0,*popt_tmp = popt
        popt1 = popt_tmp[:10]
        popt2 = popt_tmp[10:]
        first  = self.simple_fourier(x,p1,A0,*popt1) - A0
    
        phot_obj = photdata([x,y-first,yerr])
        second_component = phot_obj.fourier_composition(x,2*np.pi/p2,*phot_obj.get_best_fit_at_p(p2))
        first_component = y - second_component
        phot_obj2 = photdata([x,first_component,yerr])
        first_tmp = phot_obj2.fourier_composition(x,2*np.pi/p1,*phot_obj2.get_best_fit_at_p(p1))
        second_component = y - first_tmp - A0
    
        fig,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=figsize,sharey=True)
    
        ax1.scatter(x%p1/p1,y,s=5,c='k')
        ax1.scatter(x%p1/p1+1,y,s=5,c='k')
        ax1.set_xlabel('phase',fontsize=20)
        ax1.set_ylabel('V (mag)',fontsize=20)
        ax1.set_title(self.label+' data folded at p1',fontsize=18)
        ax1.invert_yaxis()
        ax1.set_xticks(ticks=[0,0.5,1,1.5,2])
        ylim = ax1.get_ylim()
    
        ax2.scatter(x%p2/p2,y,s=5,c='k')
        ax2.scatter(x%p2/p2+1,y,s=5,c='k')
        ax2.set_xlabel('phase',fontsize=20)
        ax2.set_title(self.label+' data folded at pf',fontsize=18)
        ax2.invert_yaxis()
        ax2.set_xticks(ticks=[0,0.5,1,1.5,2])
        ax2.set_ylim(ylim)
    
        ax3.scatter(x%p1/p1,A0+first_component,s=5,c='k')
        ax3.scatter(x%p1/p1+1,A0+first_component,s=5,c='k')
        ax3.set_title('First overtone: p1={:.7f} (d)'.format(p1),fontsize=18)
        ax3.set_ylim(ylim)
        ax3.set_xticks(ticks=[0,0.5,1,1.5,2])
        ax3.set_xlabel('phase',fontsize=20)
    
        ax4.scatter(x%p2/p2,A0+second_component,s=5,c='k')
        ax4.scatter(x%p2/p2+1,A0+second_component,s=5,c='k')
        ax4.set_title('Fundamental mode: pf={:.7f} (d)'.format(p2),fontsize=18)
        ax4.set_ylim(ylim)
        ax4.set_xticks(ticks=[0,0.5,1,1.5,2])
        ax4.set_xlabel('phase',fontsize=20)
        plt.subplots_adjust(wspace=0.05)
        if show:
            plt.show()
        return x,A0,first_component,second_component,yerr
        
    ###########################    
    ### period detection
    ###########################  
    def detect_period_quick(self,max_width=1e-7,max_iteration=100,test_num=100,threshold=0.1,initial_search_width=1e-6,convergence_size_ratio=0.03,title='',show_plot=True,show_results=True,period_guess=None):
        start_time = time.time()
        def quadratic(x,a,b,c):
            return a*(x-b)**2+c
        
        if period_guess == None:
            # initial analysis with Lomb-Scargle
            print('Starting the process with Lomb-Scargle periodogram...')
            try:
                freqs = np.array(self.get_LS()[2])
            except:
                return 0,0,0
        else:
            freqs = [[1/period_guess]]

        print('Continuing the process with chi2 potential analysis...')
        potential_min_list = []
        potential_list = []
        test_p_list_list = []
        isInitial=True
        for f in freqs:
            p = 1/f[0]
            if not isInitial:
                if (p>test_p_list[0]) and (p<test_p_list[-1]):
                    print('searching for the minimum around p={}... skipped (overlap)'.format(p))
                    continue
            else:
                isInitial=False
            print('searching for the minimum around p={}...'.format(p))
            test_p_list = np.linspace(p-1000*initial_search_width,p+1000*initial_search_width,10*test_num)
            chi2_potential = self.get_global_potential(test_p_list)
            if show_plot:
                plt.figure()
                plt.scatter(test_p_list,chi2_potential,s=4)
                plt.xlim(np.min(test_p_list),np.max(test_p_list))
                plt.ylim(np.min(chi2_potential),np.max(chi2_potential))
                plt.show()
            while (chi2_potential[0]==np.min(chi2_potential) or chi2_potential[-1]==np.min(chi2_potential)):
                print('Searching for the bottom of the potential...')
                p = test_p_list[chi2_potential==np.min(chi2_potential)][0]
                test_p_list = np.linspace(p-1000*initial_search_width,p+1000*initial_search_width,10*test_num)
                chi2_potential = self.get_global_potential(test_p_list)
                if show_plot:
                    plt.figure()
                    plt.scatter(test_p_list,chi2_potential,s=4)
                    plt.xlim(np.min(test_p_list),np.max(test_p_list))
                    plt.ylim(np.min(chi2_potential),np.max(chi2_potential))
                    plt.show()
            potential_min_list.append(np.min(chi2_potential))
            potential_list.append(chi2_potential)
            test_p_list_list.append(test_p_list)
        test_p_list_list = np.array(test_p_list_list)
        potential_min_list = np.array(potential_min_list)
        potential_list = np.array(potential_list)
        p = test_p_list_list[np.argmin(potential_min_list)][potential_list[np.argmin(potential_min_list)]==potential_min_list[np.argmin(potential_min_list)]][0]
            
        print('Potential lock: success (p={:.9f})'.format(p))
        search_width = initial_search_width
        while (search_width>max_width):
            print('Searching for the best-fit period: width={:.2e}'.format(search_width))
            test_p_list = np.linspace(p-search_width,p+search_width,test_num)
            chi2_potential = self.get_global_potential(test_p_list)
            p = test_p_list[chi2_potential==np.min(chi2_potential)][0]
            search_width *= 0.5
        

        p0 = [np.std(chi2_potential)/search_width**2,test_p_list[chi2_potential==np.min(chi2_potential)][0],np.min(chi2_potential)]
        popt,pcov = curve_fit(quadratic,test_p_list,chi2_potential,p0=p0)
        period = popt[1]
        print('Period estimation is complete. Performing final analysis:')

        test_p_list = np.linspace(period-search_width,period+search_width,test_num)
        chi2_potential = self.get_global_potential(test_p_list)
        p0 = [np.std(chi2_potential)/search_width**2,test_p_list[chi2_potential==np.min(chi2_potential)][0],np.min(chi2_potential)]
        popt,pcov = curve_fit(quadratic,test_p_list,chi2_potential,p0=p0)
        period = popt[1]
        print('')
        print('************************')
        print('* Period = {:.9f} *'.format(period))
        print('* error  = undefined (quick mode)')
        print('************************')

        # summary
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,5))

        # potential plot
        ax1.scatter(test_p_list,chi2_potential,s=8,c='k')
        ax1.plot(test_p_list,quadratic(test_p_list,*popt),color='orange',label='quadratic fit')
        ax1.axvline(period,color='green',label='period')
        ax1.set_xlim(np.min(test_p_list)-0.1*np.std(test_p_list),np.max(test_p_list)+0.1*np.std(test_p_list))
        ax1.set_ylim(np.min(chi2_potential)-0.1*np.std(chi2_potential),np.max(chi2_potential)+0.1*np.std(chi2_potential))
        ax1.legend()

        # generate best-fit light curve
        omega = 2*np.pi/period
        popt = self.get_best_fit_at_p(period)
        x_th = np.linspace(0,2*period,1000)
        y_th = self.fourier_composition(x_th,2*np.pi/period,*popt)
        ax2.scatter(self.x%period,self.y,color='k',s=8)
        ax2.scatter(self.x%period+period,self.y,color='k',s=8)
        ax2.plot(x_th,y_th)
        ax2.set_xlabel('time [day]',fontsize=15)
        ax2.set_ylabel('mag',fontsize=15)
        ax2.set_title(title + '  P={:.9f}'.format(period))
        ax2.invert_yaxis()        
        amplitude = np.max(y_th)-np.min(y_th)
        plt.show()
        self.period     = period
        self.period_err = None
        self.amplitude  = amplitude
        print('----- End of period detection. Execution time: {:.3f} seconds -----'.format(time.time()-start_time))
        return period,amplitude    
    
    def detect_period(self,max_iteration=100,test_num=100,threshold=0.1,initial_search_width=1e-6,convergence_size_ratio=0.03,title='',show_plot=True,show_results=True,period_guess=None):
        start_time = time.time()
        def quadratic(x,a,b,c):
            return a*(x-b)**2+c
        
        if period_guess == None:
            # initial analysis with Lomb-Scargle
            print('Starting the process with Lomb-Scargle periodogram...')
            try:
                freqs = np.array(self.get_LS()[2])
            except:
                return 0,0,0
        else:
            freqs = [[1/period_guess]]

        print('Continuing the process with chi2 potential analysis...')
        # Phase zero: potential lock
        # first move around until the bottom of the potential is found
        # this phase is added in case Lomb-Scargle's estimation is way off from the true value.
        potential_min_list = []
        potential_list = []
        test_p_list_list = []
        isInitial=True
        for f in freqs:
            p = 1/f[0]
            if not isInitial:
                if (p>test_p_list[0]) and (p<test_p_list[-1]):
                    print('searching for the minimum around p={}... skipped (overlap)'.format(p))
                    continue
            else:
                isInitial=False
            print('searching for the minimum around p={}...'.format(p))
            test_p_list = np.linspace(p-1000*initial_search_width,p+1000*initial_search_width,10*test_num)
            chi2_potential = self.get_global_potential(test_p_list)
            if show_plot:
                plt.figure()
                plt.scatter(test_p_list,chi2_potential,s=4)
                plt.xlim(np.min(test_p_list),np.max(test_p_list))
                plt.ylim(np.min(chi2_potential),np.max(chi2_potential))
                plt.show()
            while (chi2_potential[0]==np.min(chi2_potential) or chi2_potential[-1]==np.min(chi2_potential)):
                print('Searching for the bottom of the potential...')
                p = test_p_list[chi2_potential==np.min(chi2_potential)][0]
                test_p_list = np.linspace(p-1000*initial_search_width,p+1000*initial_search_width,10*test_num)
                chi2_potential = self.get_global_potential(test_p_list)
                if show_plot:
                    plt.figure()
                    plt.scatter(test_p_list,chi2_potential,s=4)
                    plt.xlim(np.min(test_p_list),np.max(test_p_list))
                    plt.ylim(np.min(chi2_potential),np.max(chi2_potential))
                    plt.show()
            potential_min_list.append(np.min(chi2_potential))
            potential_list.append(chi2_potential)
            test_p_list_list.append(test_p_list)
        test_p_list_list = np.array(test_p_list_list)
        potential_min_list = np.array(potential_min_list)
        potential_list = np.array(potential_list)
        p = test_p_list_list[np.argmin(potential_min_list)][potential_list[np.argmin(potential_min_list)]==potential_min_list[np.argmin(potential_min_list)]][0]
            
        print('Potential lock: success (p={:.9f})'.format(p))
        test_p_min = p-initial_search_width
        test_p_max = p+initial_search_width
        test_p_list,p_list,chisq_list = self.test_global_potential(test_p_min,test_p_max,test_num)
        chi2_potential = self.get_global_potential(test_p_list)
        if show_plot:
            self.potential_plot(test_p_list,p_list,chisq_list,chi2_potential)
        previous_std = min(np.std(abs(test_p_list-p_list)),np.std(p_list))
        final_check_flag = False
        detection = False
        previous_width = test_p_max - test_p_min
        # Phase I: best-fit-chi2-based
        # This method selects only one potential when there are multiple local minima.
        # Same result can be achieved by gradually narrowing the search window, but this method is faster
        # especially when the width of the peak is much narrower than the initial search width.
        print('Repeating the process with smaller search width...')
        for _ in range(max_iteration):
            # new search range
            chisq_cutline = np.min(chisq_list) + threshold*(np.max(chisq_list)-np.min(chisq_list))
            test_p_min_tmp = np.min(test_p_list[chisq_list < chisq_cutline])
            test_p_max_tmp = np.max(test_p_list[chisq_list < chisq_cutline])
            if (test_p_max_tmp-test_p_min_tmp)/2 <= 2*previous_std:
                print('Phase I is at the limit of its capability (error I-01). Proceeding to Phase II...')
                break
            elif test_p_min_tmp > test_p_list[chi2_potential==np.min(chi2_potential)][0]:
                print('Phase I is at the limit of its capability (error I-02). Proceeding to Phase II...')
                break
            elif test_p_max_tmp < test_p_list[chi2_potential==np.min(chi2_potential)][0]:
                print('Phase I is at the limit of its　capability (error I-03). Proceeding to Phase II...')
                break
            else:
                test_p_min = test_p_min_tmp
                test_p_max = test_p_max_tmp
            # do analysis
            test_p_list,p_list,chisq_list = self.test_global_potential(test_p_min,test_p_max,test_num)
            chi2_potential = self.get_global_potential(test_p_list)
            if show_plot:
                self.potential_plot(test_p_list,p_list,chisq_list,chi2_potential)

            # did std converge?
            current_std = min(np.std(abs(test_p_list-p_list)),np.std(p_list))
            print('old std = {}'.format(previous_std))
            print('new std = {}'.format(current_std))
        ### TODO: clean this up ###
            current_width = test_p_max - test_p_min
            if abs(1-current_std/previous_std) >= convergence_size_ratio or test_p_max - test_p_min < 0.8* previous_width:
                print('std changed by {:.1f}% and width changed by {:.1f}%. Repeating the process with smaller search width...'.format(abs(1-current_std/previous_std)*100,abs(1-current_width/previous_width)*100))
                previous_std = current_std
                previous_width = current_width
            else:
                print('std changed by {:.1f}%, which is within the specified torelance.'.format(abs(1-current_std/previous_std)*100))
                print('Finished phase I. Proceeding to phase II...')
                previous_std = current_std
                break

        # Phase II: global_potential_based
        # This method determines the peak as well as the standard deviation of the period estimation.
        # std is max(statistical,systematic) 
        # where statistic  = min( std(abs(guess-estimated)) , std(estimated) ), and
        #       systematic = sqrt(diag(cov_matrix))[idx_period] from the quadratic fit.
        # systematic error is only calculated in Phase III, so this phase only considers statistic error.
        width = 0.8*(test_p_max - test_p_min)/2
        for _ in range(max_iteration):
            # update search width based on quadratic fit
            period = test_p_list[chi2_potential==np.min(chi2_potential)][0]
            test_p_min = period - width
            test_p_max = period + width

            # do analysis
            test_p_list,p_list,chisq_list = self.test_global_potential(test_p_min,test_p_max,test_num)
            chi2_potential = self.get_global_potential(test_p_list)
            if show_plot:
                self.potential_plot(test_p_list,p_list,chisq_list,chi2_potential)

            # did std converge?
            current_std = min(np.std(abs(test_p_list-p_list)),np.std(p_list))
            print('old std = {}'.format(previous_std))
            print('new std = {}'.format(current_std))
            if abs(1-current_std/previous_std) >= convergence_size_ratio:
                print('std changed by {:.1f}%. Repeating the process with 20% smaller search width...'.format(abs(1-current_std/previous_std)*100))
                previous_std = current_std
                width *= 0.8
            else:
                print('std changed by {:.1f}%, which is within the specified torelance.'.format(abs(1-current_std/previous_std)*100))
                print('Finished phase II.')
                previous_std = current_std
                detection = True
                break

        # Phase III: summary
        # fit again. replaces std with systematic if the fitting is not so good.
        if detection == True:
            p0 = [np.std(chi2_potential)/current_std**2,test_p_list[chi2_potential==np.min(chi2_potential)][0],np.min(chi2_potential)]
            popt,pcov = curve_fit(quadratic,test_p_list,chi2_potential,p0=p0)
            period = popt[1]
            print('Period estimation is complete. Performing final analysis:')

            # zoom out if we have gotten too close during Phase II (at least 5 sigma)
            if (test_p_max-test_p_min)/2 < 5*current_std:
                test_p_min = period - 5*current_std
                test_p_max = period + 5*current_std
                test_p_list = np.linspace(p-initial_search_width,p+initial_search_width,test_num)

            # zoom in if possible: we tend to get cleaner quadratic fit near the peak (at most 10 sigma)
            if (test_p_max-test_p_min)/2 > 10*current_std:
                print('Zooming in to the peak...')
                for scale in [100,50,10]:
                    test_p_min = period - scale*current_std
                    test_p_max = period + scale*current_std
                    test_p_list = np.linspace(test_p_min,test_p_max,test_num)
                    chi2_potential = self.get_global_potential(test_p_list)
                    while (chi2_potential[0]==np.min(chi2_potential) or chi2_potential[-1]==np.min(chi2_potential)):
                        period = test_p_list[chi2_potential==np.min(chi2_potential)][0]
                        test_p_min = period - scale*current_std
                        test_p_max = period + scale*current_std
                        test_p_list = np.linspace(test_p_min,test_p_max,test_num)
                        chi2_potential = self.get_global_potential(test_p_list)
                    period = test_p_list[chi2_potential==np.min(chi2_potential)][0]
                    test_p_min = period - scale*current_std
                    test_p_max = period + scale*current_std
                    test_p_list = np.linspace(test_p_min,test_p_max,test_num)
                    chi2_potential = self.get_global_potential(test_p_list)
                period = test_p_list[chi2_potential==np.min(chi2_potential)][0]
                test_p_min = period - scale*current_std
                test_p_max = period + scale*current_std        

            test_p_list,p_list,chisq_list = self.test_global_potential(test_p_min,test_p_max,test_num)
            chi2_potential = self.get_global_potential(test_p_list)
            p0 = [np.std(chi2_potential)/current_std**2,test_p_list[chi2_potential==np.min(chi2_potential)][0],np.min(chi2_potential)]
            popt,pcov = curve_fit(quadratic,test_p_list,chi2_potential,p0=p0)
            period = popt[1]
            std = max(current_std,np.sqrt(np.diag(pcov))[1])
            print('')
            print('************************')
            print('* Period = {:.9f} *'.format(period))
            print('* error  = {:.9f} *'.format(std))
            print('************************')


            fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,5))

            # potential plot
            ax1.scatter(test_p_list,chi2_potential,s=8,c='k')
            ax1.plot(test_p_list,quadratic(test_p_list,*popt),color='orange',label='quadratic fit')
            ax1.axvline(period,color='green',label='period')
            ax1.axvspan(period-std,period+std,color='green',alpha=0.2,label='uncertainty')
            ax1.set_xlim(np.min(test_p_list)-0.1*np.std(test_p_list),np.max(test_p_list)+0.1*np.std(test_p_list))
            ax1.set_ylim(np.min(chi2_potential)-0.1*np.std(chi2_potential),np.max(chi2_potential)+0.1*np.std(chi2_potential))
            ax1.legend()

            # generate best-fit light curve
            omega = 2*np.pi/period
            popt = self.get_best_fit_at_p(period)
            x_th = np.linspace(0,2*period,1000)
            y_th = self.fourier_composition(x_th,2*np.pi/period,*popt)
            ax2.scatter(self.x%period,self.y,color='k',s=8)
            ax2.scatter(self.x%period+period,self.y,color='k',s=8)
            ax2.plot(x_th,y_th)
            ax2.set_xlabel('time [day]',fontsize=15)
            ax2.set_ylabel('mag',fontsize=15)
            ax2.set_title(title + '  P={:.9f}({:})'.format(period,int(std/1e-9)))
            ax2.invert_yaxis()        
            amplitude = np.max(y_th)-np.min(y_th)
            plt.show()
        else:
            print('Unable to determine period within given iteration number. Increase the trial size and try again.')
            return 0,0,0
        self.period     = period
        self.period_err = std
        self.amplitude  = amplitude
        print('----- End of period detection. Execution time: {:.3f} seconds -----'.format(time.time()-start_time))
        return period,std,amplitude

    def detect_secondperiod(self,initial_search_width=1e-5,informed=True,p1_manual=None,p2_manual=None):
        popt,period = self.fit_doublemode_initial(period_manual=p1_manual)
        if period < 0.5:
            p2,A0,A1,A2,phi1,phi2 = popt
            p1 = period
            period_to_search = p2
        else:
            p1,A0,A1,A2,phi1,phi2 = popt
            p2 = period        
            period_to_search = p1
        x = self.x
        y = self.y
        yerr = self.yerr
        first_component = A1*np.sin(2*np.pi/p1*x - phi1)
    
        # 1st fitting
        phot_obj = photdata([x,y-first_component,yerr])
        if p2_manual != None:
            p2 = p2_manual
        if informed == True:
            phot_obj.detect_period_quick(max_width=1e-7,show_plot=False,period_guess=period_to_search,initial_search_width=initial_search_width)
        else:
            phot_obj.detect_period_quick(max_width=1e-7,show_plot=False,initial_search_width=initial_search_width)
            
        if period < 0.5:
            p1 = period
            p2 = phot_obj.period
        else:
            p1 = phot_obj.period
            p2 = period
        self.p1=p1
        self.p2=p2
        print('p1 = {:.7f}'.format(p1))
        print('pf = {:.7f}'.format(p2))
        print('p1/pf = {:.3f}'.format(p1/p2))
        return p1,p2
