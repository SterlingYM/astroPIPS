import numpy as np
from PIPS.class_photdata import photdata
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def group_data(star,separation=100):
    """ Automatically groups the data from a single photometric time series.
    
    Args:
        star (PIPS.photdata): the object that contains photometric data over a long baseline.
        separation (float): the minimum separation in the time values (e.g., days) between subsets of data contained in the provided data.
        
    Returns:
        photdata_list (list): a list of photdata. Represents the subset of data separated from the original single photdata.
    """
    sep = star.x[1:]-star.x[:-1]
    split_idx = np.where(sep >= separation)[0]

    photdata_list = []
    idx_prev = 0
    for idx in split_idx:
        idx += 1
        x_new = star.x[idx_prev:idx]
        y_new = star.y[idx_prev:idx]
        yerr_new = star.yerr[idx_prev:idx]
        photdata_list.append(photdata([x_new,y_new,yerr_new]))
        idx_prev = idx
    return photdata_list

class longdata:
    def __init__(self,photdata_list):
        ''' Inititalizes the longdata object.
        
        Args:
            photdata_list (list): a list of photdata objects each of which contains the photometric values in the same time unit.
        '''
        self.data = photdata_list
        self.periods = []
        self.x_means = []
        self.oc = []
        self.oc_err = []
        self.p_ref = []
        self.y_offsets = []
        self.alphas = []
        self.betas = []
        self.model_popt = None

    def __len__(self):
        return len(self.data)

    def prep_alldata(self):
        """ prepares the concatenated data.
        """
        all_data = photdata([[],[],[]])
        for star in self.data:
            all_data = all_data + star
            self.x_means.append(star.x.mean())
        self.all_data = all_data

    def run_OC(self,
                p_ref = None, p_ref_err = None,
                template_idx = None, 
                method = 'fast', model = 'Fourier',
                p0_func = None, round_mode = 'round',
                **kwargs):
        """ Calculates the O-C values for given data.
        
        Args: 
            p_ref: the reference period value.
            template_idx: the index of the dataset that points to the "template" data chunk
            method (str)['fast','custom']: see the documentation of photdata._get_period().
            model (str/obj): see the documentation of photdata._get_period().
            p0_func (obj): see the documentation of photdata._get_period().
            round_mode (str)['round','floor','ceil']: the lower and upper limits of O-C values.
            
        Returns:
            self: the longdata object. Attributes such as self.oc or self.oc_err contains the values associated with each subset of data.
        """
        # generate concatenated data
        self.prep_alldata()

        # prepare 'mean' curve
        if template_idx is None:
            print('template is not specified: using the first photdata as template')
            template_idx = 0
        template_star = self.data[template_idx]
        if p_ref is None:
            p_ref,p_ref_err = template_star.get_period(**kwargs)
            print(p_ref,p_ref_err)
        model_popt = template_star.get_bestfit_curve(
            period=p_ref,
            return_params=True,
            model='Fourier',
            p0_func=None,
            **kwargs)
        self.model_popt = model_popt

        # # prepare 'mean' period
        # if p_ref is None:
        #     print('p_ref not given: using the template period as p_ref')
        #     p_ref = template_period
            
        # select models
        if method=='fast':
            if 'Nterms' in kwargs:
                Nterms = kwargs['Nterms']
            else:
                kwargs['Nterms'] = 5 
        MODEL, P0_FUNC, KWARGS = self.all_data.check_model(model, p0_func, kwarg_for_helper=True,**kwargs)

        # get O-C by fitting offsets
        for star in self.data:
            if star.period is None:
                try:
                    star.get_period(**kwargs)
                except Exception:
                    star.period=np.nan

            if round_mode=='round':
                bounds = ([-p_ref/2,-np.inf],[p_ref/2,np.inf])
                p0 = [0,0]
            elif round_mode=='floor':
                bounds = ([0,-np.inf],[p_ref,np.inf])
                p0 = [p_ref/2,0]
            elif round_mode=='ceil':
                bounds = ([-p_ref,-np.inf],[0,np.inf])
                p0 = [-p_ref/2,0]
            else:
                raise ValueError(f'incorrect round_mode: {round_mode} not in '+'{\'round\',\'floor\',\'ceil\'}')
            popt,pcov = curve_fit(
                lambda x,mu,nu: MODEL(x-mu,p_ref,model_popt,**KWARGS)+nu,
                star.x,
                star.y,
                sigma=star.yerr,
                p0=p0,
                bounds=bounds
            )

            # statistical error
            OC_err_stat = np.sqrt(np.diag(pcov))[0]

            # estimate the systematic error due to the choice of p_ref
            baseline = star.x.mean() - template_star.x.mean() 
            OC_err_syst = p_ref_err * baseline / p_ref

            self.periods.append(star.period)
            self.oc.append(popt[0])
            self.oc_err.append(np.sqrt(OC_err_stat**2 + OC_err_syst**2))
            self.y_offsets.append(popt[1])
        self.p_ref = p_ref
        self.p_ref_err = p_ref_err
        return self

    def plot_oc(self,ax=None,figsize=(8,5),return_axis=False,**kwargs):
        """ plot the O-C diagram.
        
        Args:
            ax (matplotlib.axes.Axes): the user-defined axis.
            figsize (tuple): the figure size.
            return_axis (bool): the option to return the axis object after plotting.        
        
        Returns:
            ax (matplotlib.axis.Axes): the axis.  
        """
        
        if ax==None:
            fig,ax = plt.subplots(1,1,figsize=(8,5))
        ax.errorbar(self.x_means,self.oc,self.oc_err,fmt='o',**kwargs)
        if return_axis:
            return ax

    def plot_lc(self,ax=None,figsize=(8,5),return_axis=False,invert_yaxis=True,**kwargs):
        """ plots the phase-folded lightcurve to highlight the phase shift over time.
        
        Args:
            ax (matplotlib.axes.Axes): the user-defined axis.
            figsize (tuple): the figure size.
            return_axis (bool): the option to return the axis object after plotting.
            invert_yaxis (bool): the option to invert the y-axis. True by default assuming the magnitude scale.
        
        Returns:
            ax (matplotlib.axis.Axes): the axis.        
        """
        if ax==None:
            fig,ax = plt.subplots(1,1,figsize=(8,5))

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        for star,color in zip(self.data,colors):
            ax.errorbar(star.x%self.p_ref,star.y,star.yerr,fmt='o',ms=1,label=star.label,color=color,**kwargs)
            ax.errorbar(star.x%self.p_ref+self.p_ref,star.y,star.yerr,fmt='o',ms=1,color=color,**kwargs)
        ax.legend()
        if invert_yaxis:
            ax.invert_yaxis()
        if return_axis:
            return ax


        
        
