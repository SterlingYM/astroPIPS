import numpy as np
from ..class_photdata import photdata
from scipy.optimize import curve_fit

def group_data(star,separation=100):
    sep = star.x[1:]-star.x[:-1]
    split_idx = np.where(sep>=separation)[0]

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
        all_data = photdata([[],[],[]])
        for star in self.data:
            all_data = all_data + star
            self.x_means.append(star.x.mean())
        self.all_data = all_data

    def run_OC(self,period=None,method='fast',model='Fourier',p0_func=None,round_mode='round',**kwargs):
        
        # generate concatenated data
        self.prep_alldata()

        # prepare 'mean' period
        if period==None:
            period,_ = self.all_data.get_period(**kwargs)
        else:
            self.all_data.period = period
        self.p_ref = period

        # prepare 'mean' curve
        model_popt = self.all_data.get_bestfit_curve(period=period,return_params=True,model='Fourier',p0_func=None,**kwargs)
        self.model_popt = model_popt

        # select models
        if method=='fast':
            if 'Nterms' in kwargs:
                Nterms = kwargs['Nterms']
            else:
                kwargs['Nterms'] = 5 
        MODEL, P0_FUNC, KWARGS = self.all_data.check_model(model, p0_func, kwarg_for_helper=True,**kwargs)

        # get O-C by fitting offsets
        for star in self.data:
            if star.period == None:
                try:
                    star.get_period(**kwargs)
                except Exception:
                    star.period=np.nan

            if round_mode=='round':
                bounds = [[-period/2,-np.inf],[period/2,np.inf]]
            elif round_mode=='floor':
                bounds = [[0,-np.inf],[period,np.inf]]
            elif round_mode=='ceil':
                bounds = [[-period,-np.inf],[0,np.inf]]
            else:
                raise ValueError(f'incorrect round_mode: {round_mode} not in '+'{\'round\',\'floor\',\'ceil\'}')
            popt,pcov = curve_fit(
                lambda x,mu,nu: MODEL(x-mu,period,model_popt,**KWARGS)+nu,
                star.x,
                star.y,
                sigma=star.yerr,
                bounds=bounds
            )
            self.periods.append(star.period)
            self.oc.append(popt[0])
            self.oc_err.append(np.sqrt(np.diag(pcov))[0])
            self.y_offsets.append(popt[1])



        
        