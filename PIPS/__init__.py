'''
-----------
PIPS 0.3.0
---------------------------------------------
Developers: Y. Murakami, A. Hoffman, J.Sunseri, A. Savel
Contact: Yukei Murakami (sterling.astro@berkeley.edu)
License: MIT
---------------------------------------------
Processes photometric data for variable stars.
---------------------------------------------

Classes:
    photdata  --- data container for individual objects and analysis tools
    visualize --- visualization tools for photdata and analysis results
    StellarModels --- various stellar property relationship models (e.g. period-luminosity)

Independent Functions:
    get_bestfit_Fourier(x,y,yerr,period,return_yfit=True,return_params=False)
    get_bestfit_GM(x,y,yerr,period,return_yfit=True,return_params=False)
    get_chi2_Fourier(x,y,yerr,period,Nterms=4)
    get_chi2ref(x,y,yerr)
    Fourier(self,period,params)
    OC(photdata_obj,)
'''
import matplotlib.pyplot as pyplot
import numpy as np
from scipy.optimize import curve_fit
import numba
from multiprocessing import Pool


from .utils.connect_LPP import data_readin_LPP
from .class_photdata import photdata
from .class_StellarModels import *
# from .class_visualize import visualize


__uri__ = "https://PIPS.readthedocs.io" # tbd, may change
__author__ = "Y. Murakami, A. Hoffman, J.Sunseri, A. Savel"
__maintainer__ = "Y. Murakami"
__email__ = "sterling.astro@berkeley.edu"
__license__ = "MIT"
__version__ = "0.3.0"
__release__ = "0.3.0-alpha"
__description__ = "Processes photometric data for variable stars"
