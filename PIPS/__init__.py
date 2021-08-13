'''
-----------
PIPS 0.3.0
---------------------------------------------
Developers: Y. Murakami, A. Savel, J. Sunseri, A. Hoffman
Contact: Yukei Murakami (sterling.astro@berkeley.edu)
License: MIT
---------------------------------------------
Processes photometric data for variable stars.
---------------------------------------------

Classes:
    photdata  --- data container for individual objects and analysis tools
    visualize --- visualization tools for photdata and analysis results
    StellarModels --- various stellar property relationship models (e.g. period-luminosity)
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


__uri__ = "https://PIPS.readthedocs.io" 
__author__ = "Y. Murakami, A. Savel, J. Sunseri, A. Hoffman"
__maintainer__ = "Y. Murakami"
__email__ = "sterling.astro@berkeley.edu"
__license__ = "MIT"
__version__ = "0.3.0-alpha.9"
__release__ = "0.3.0-alpha.9"
__description__ = "Processes photometric data for variable stars"

def about():
    text =  "--------------------------\n"
    text += "-    Welcome to PIPS!    -\n"
    text += "--------------------------\n"
    text += "Version: " + __version__ + '\n'
    text += "Authors: " + __author__ + '\n'
    text += "--------------------------\n"
    text += "Download the latest version from: https://pypi.org/project/astroPIPS\n"
    text += "Report issues to: https://github.com/SterlingYM/astroPIPS\n"
    text += "Read the documentations at: " + __uri__ + '\n'
    text += "--------------------------\n"
    print(text)

if __name__ == '__main__':
    about()