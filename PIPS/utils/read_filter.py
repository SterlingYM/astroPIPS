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
    with pkg_resources.path(filters,filename) as p:
        wav,res = np.loadtxt(p,unpack=True,delimiter=delim)
    return wav,res

def prep_filterdata(filenames,filternames,delim=None):
    filterdata = {}
    for file,filter in zip(filenames,filternames):
        filterdata.update({filter:read_filter(file,delim)})
    return filterdata