import numpy as np
def data_readin_LPP(path,filter='V'):
    '''
    takes '.dat' file from LOSS Phot Pypeline (LPP) and returns data in PIPS.photdata()-readable format.
    
    Args:
        path: (str) full path to file to be read in.
        filter: (str, optional) filter of data to be read in from the file.
        
    Returns:
        data: (list of np.array) time, y, and yerr read in from LPP file.
    '''
    # load info
    t,y,y_lower,y_upper = np.loadtxt(path,delimiter='\t',usecols=(0,2,3,4),skiprows=1,unpack=True)
    band                = np.loadtxt(path,delimiter='\t',usecols=6,skiprows=1,dtype=str,unpack=True)

    # uncertainty is not linear in log scale (mag) so (y_upper-y) != (y-y_lower).
    # taking the average of these two is not perfectly accurate, but it works (TODO: modify this?)
    yerr = (y_upper - y_lower)/2 
    
    # separate into specified band
    data = [t[band==filter],y[band==filter],yerr[band==filter]]
    return data
