import numpy as np
import matplotlib.pyplot as plt
import pickle
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
import pandas as pd


def plot_all(data):
    for i in range(len(data)):
        plt.figure(figsize=(10,3))
        x,y,yerr = data[i]
        plt.scatter(x,y)

        # yrange
        mean = np.nanmean(y)
        plt.ylim(mean-1,mean+1)
        plt.grid()
        plt.show()
    
def plot_single(data,i,ax=None):
    if ax==None:
        fig,ax = plt.subplots(1,1,figsize=(10,3))
    x,y,yerr = data[i]
    ax.errorbar(x,y,yerr=yerr,fmt='o',mfc='black',mec='none',ms=4)#,s=15,color='black')
    
    # yrange
    mean = np.nanmean(y)
    ax.set_ylim(mean-1,mean+1)
    ax.set_xlabel('time [days]')
    ax.set_ylabel('mag')
    ax.set_title('M15_1 star # {}'.format(i))
    ax.invert_yaxis()
    ax.grid()

def get_noNaN(data):
    # remove NaN (Not a Number) data points
    data_nonan = []
    for i in range(len(data)):
        xdata,ydata,yerr = data[i]
        condition = np.all([np.isfinite(xdata),np.isfinite(ydata),np.isfinite(yerr)],axis=0)
        xdata = np.array(xdata)
        ydata = np.array(ydata)
        yerr  = np.array(yerr)
        x_nonan = xdata[condition]
        y_nonan = ydata[condition]
        yerr_nonan = yerr[condition]
        data_nonan.append([x_nonan,y_nonan,yerr_nonan])
    return data_nonan

def get_period(data,idx):
    x,y,err = data[idx]
    freq,power=LombScargle(x,y).autopower(minimum_frequency=1,maximum_frequency=10.0)
    p_sorted = np.sort(power)
    f0 = freq[power==p_sorted[-1]]
    f1 = freq[power==p_sorted[-2]]
    f2 = freq[power==p_sorted[-3]]
    f3 = freq[power==p_sorted[-4]]    
    return freq,power,[f0,f1,f2,f3]

def get_period_random(data,idx,which='y'):
    x,y,err = data[idx]
    x1 = np.array(x)
    y1 = np.array(y)
    if which=='y':
        np.random.shuffle(y1)
    elif which=='x':
        np.random.shuffle(x1)
    freq,power=LombScargle(x1,y1).autopower(minimum_frequency=0.1,maximum_frequency=10.0)
    pf = freq[power==np.max(power)]
    return freq,power,pf

def get_component(data,idx,freq=None,power=None):
    # TODO: improve this
    x,y,err = data[idx]
    if np.all(freq) == None or np.all(power) == None:
        freq,power,_=get_period(data,idx)
        p_sorted = np.sort(power)
        f0 = freq[power==p_sorted[-1]]
        f1 = freq[power==p_sorted[-2]]
        f2 = freq[power==p_sorted[-3]]
        f3 = freq[power==p_sorted[-4]]
    else:
        f0=0;f1=0;f2=0;f3=0;p_sorted=[0,0,0,0]
    return [f0,f1,f2,f3],[p_sorted[0],p_sorted[1],p_sorted[2],p_sorted[3]]
    
def plot_period(data,idx,ax=None,power_lower_lim=None):
    freq,power,freqs = get_period(data,idx)
    pf = freqs[0]
    f_list,p_list = get_component(data,idx)
    
    if ax == None:
        fig,ax = plt.subplots(1,1,figsize=(10,3))
    if not np.all(np.isnan(power)):
        ax.plot(freq,power,
               alpha = 1 if (power_lower_lim == None) \
                or (power[freq==pf]>=power_lower_lim) else 0.3)
        for f in f_list:
            ax.axvline(f,color='red',alpha=0.1)
    ax.set_title('Lomb-Scargle Periodogram')
    ax.set_ylabel('power')
    ax.set_xlabel('frequency [1/days]')
    ax.set_ylim(0,1)
    return freq,power,pf

def plot_period_random(data,idx,ax=None,power_lower_lim=None,which='y'):
    freq,power,pf = get_period_random(data,idx,which=which)
    f_list,p_list = get_component(data,idx,freq=freq,power=power)
    
    if ax == None:
        fig,ax = plt.subplots(1,1,figsize=(10,3))
    if not np.all(np.isnan(power)):
        ax.plot(freq,power,
               alpha = 1 if (power_lower_lim == None) \
                or (power[freq==pf]>=power_lower_lim) else 0.3)
        for f in f_list:
            ax.axvline(f,color='red',alpha=0.1)
    ax.set_title('Periodogram with randomized sample: '+which)
    ax.set_ylabel('power')
    ax.set_xlabel('frequency [1/days]')
    ax.set_ylim(0,1)
    return freq,power,pf

def randomize_test(data):
    for i in range(len(data)):
        fig,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(20,2.5))
        plot_single(data,i,ax=ax1)
        plot_period(data,i,ax=ax2,power_lower_lim=0.3)
        plot_period_random(data,i,ax=ax3,power_lower_lim=0.4,which='x')
        plot_period_random(data,i,ax=ax4,power_lower_lim=0.4,which='y')
        plt.show()


## internally called function
def fit_function(t,omega,N,A0,*ab_list):
    k_list = np.array(range(N))+1
    a_list = ab_list[:N]
    b_list = ab_list[N:]    
    return_val = A0
    for k,a,b in zip(k_list,a_list,b_list):
        return_val += a*np.sin(k*omega*t) + b*np.cos(k*omega*t)
    return return_val

## internally called function
def do_fitting(x,y,N,main_omega):
    A0 = np.mean(y)
    ab = np.full(2*N,1)
    p0=[A0,*ab]
    def fit_wrapper(t,A0,*ab_list):
        return fit_function(t,main_omega,N,A0,*ab_list)
    popt,pcov = curve_fit(fit_wrapper,x,y,p0=p0,maxfev=10000)
    return popt,pcov

def period_analysis(data,K,show_plot=True):
    variable_star_list = []
    for i in range(len(data)):
        freq,power,freqs = get_period(data,i)
        xdata,ydata,yerr = data[i]

        freq_sizecheck = pd.DataFrame(freqs)
        if freq_sizecheck.size >= 1 and power[freq==freqs[0]]>0.4:
            pf = freqs[0]
            if show_plot:
                fig,(ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(1,6,figsize=(20,2.5))
                plot_single(data,i,ax=ax1)
                plot_period(data,i,ax=ax2,power_lower_lim=0.4)
            xdata = np.array(xdata)
            ydata = np.array(ydata)
            yerr  = np.array(yerr)

            chi2dof_list = []
            for ax_curve,freq in zip([ax3,ax4,ax5,ax6],freqs):
                # prepare data
                x_tmp = xdata%(1/freq)
                phased_x = x_tmp[np.argsort(x_tmp)]
                phased_y = ydata[np.argsort(x_tmp)]
                phased_yerr = yerr[np.argsort(x_tmp)]
                omega = 2*np.pi*freq

                # fitting
                popt_phased,pcov_phased = do_fitting(phased_x,phased_y,K,omega)
                x_th_phased = np.linspace(np.min(phased_x),np.max(phased_x),100)
                y_th_phased = fit_function(x_th_phased,omega,K,*popt_phased)

                # calculate chi square
                expected_y  = fit_function(phased_x,omega,K,*popt_phased)
                chisq = np.sum((expected_y-phased_y)**2/yerr**2)
                chi2dof_list.append(chisq/len(phased_x))

                # plot
                if show_plot:
                    ax_curve.scatter(phased_x,phased_y,s=15,color='black')
                    ax_curve.invert_yaxis()
                    ax_curve.plot(x_th_phased,y_th_phased,'red')
                    ax_curve.set_xlabel('time [day]')
                    ax_curve.set_ylabel('mag')
                    ax_curve.set_title(r"P={:.3f}, $\chi^2/N$={:.2f}".format(float(1/freq),chisq/len(phased_x)))
                    ax_curve.grid()
            # determine the period (which makes the smallest chi2 value)
            if show_plot:
                [ax3,ax4,ax5,ax6][np.argmin(chi2dof_list)].set_facecolor("honeydew")
            chosen_period = 1/(freqs[np.argmin(chi2dof_list)])
            variable_star_list.append([xdata,ydata,yerr,chosen_period])
        plt.show()
    return variable_star_list
