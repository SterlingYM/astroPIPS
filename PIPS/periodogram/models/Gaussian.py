import numpy as np
from scipy.optimize import curve_fit
import numba
from multiprocessing import Pool

def get_bestfit_GM(x,y,yerr,period,return_yfit=True,return_params=False):
    '''
    ### Gaussian Mixture Model ###
    returns 
    '''