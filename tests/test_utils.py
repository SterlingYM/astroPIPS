import unittest
import numpy as np
import PIPS


class TestLPP(unittest.TestCase):
    
    def test_data_readin_LPP_regression_x(self):
        """
        Regression test against previously read-in 
        LPP data.
        """
        data = PIPS.data_readin_LPP('sample_data/005.dat',filter='V')
        x,y,yerr = data
        
        # read in the previous data
        x_test = np.loadtxt('tests/test_lpp_x.txt')
        np.testing.assert_array_equal(x, x_test)
        
        
    def test_data_readin_LPP_regression_y(self):
        """
        Regression test against previously read-in 
        LPP data.
        """
        data = PIPS.data_readin_LPP('sample_data/005.dat',filter='V')
        x,y,yerr = data
        
        # read in the previous data
        y_test = np.loadtxt('tests/test_lpp_y.txt')
        np.testing.assert_array_equal(y, y_test)
        
        
    def test_data_readin_LPP_regression_yerr(self):
        """
        Regression test against previously read-in 
        LPP data.
        """
        data = PIPS.data_readin_LPP('sample_data/005.dat',filter='V')
        x,y,yerr = data
        
        # read in the previous data
        yerr_test = np.loadtxt('tests/test_lpp_yerr.txt')
        np.testing.assert_array_equal(yerr, yerr_test)
        
        