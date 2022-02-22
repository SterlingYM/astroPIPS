import numpy as np
import pandas as pd
import importlib.resources as pkg_resources
from ..resources import filters

nickel2_filters = [
    [filt+'_nickel2.txt' for filt in ['B','V','R','I']],
    ['B','V','R','I']
]

kait4_filters = [
    [filt+'_kait4.txt' for filt in ['B','V','R','I']],
    ['B','V','R','I']
]

def read_filter(filename,delim=None):
    """
    Reads in data from filename according to filter.
    
    Args:
        filename: (str) full path to file to be read from.
        delim: (str or NoneType) file delimiter.
        
    Returns:
        wav: (array) wavelength array [Angstrom]
        res: (array) response function
    """
    with pkg_resources.path(filters,filename) as p:
        wav, res = np.loadtxt(p,unpack=True,delimiter=delim)
    return wav, res

def prep_filterdata(filenames,filternames,delim=None):
    """
    Prepared filterdata.
    
    Args:
        filenames: (list) list of full string paths to filenames.
        filternames: (list) list of filters corresponding to filenames. Same length
                        as filenames.
    Returns:
        filterdata: (dict) dictionary with keys as filter and values as wav,
                        res for each filename / combination as output by read_filter
    """
    filterdata = {}
    for file,filter in zip(filenames,filternames):
        filterdata.update({filter:read_filter(file,delim)})
    return filterdata
