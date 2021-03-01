import unittest
import numpy as np
from PIPS import photdata
import PIPS
import os


class TestFourier(unittest.TestCase):
    
    def test_zero_terms(self):
        """
        Should just be a flat line.
        """
        params = np.array([2,1,2, 2]) # arbitrary params
        x = np.linspace(0, 5, 400) # larger than 1 period by far
        Nterms = 0
        period = 1 # arbitrary
        y = PIPS.periodogram.models.Fourier.fourier(x, period, Nterms, params, debug=True) #
        self.assertTrue(np.all(y==params[0]))
        
    def test_one_term(self):
        """
        Should just be a cosine term.
        """
        params = np.array([2,1,2, 2]) # arbitrary params
        x = np.linspace(0, 5, 400) # larger than 1 period by far
        Nterms = 1 # just a sine term
        period = 1 # arbitrary
        y = PIPS.periodogram.models.Fourier.fourier(x, period, Nterms, params) #
        test_y = np.cos(2 * np.pi * x + params[2]) + params[0]
        np.testing.assert_array_equal(y, test_y)
        
    def test_simple_fit(self):
        x = np.linspace(0, 100, 1000)
        y = np.sin(x/2)
        yerr = np.ones_like(y) * .01

        period = 4 * np.pi

        Nterms = 1

        y_fit = PIPS.periodogram.models.Fourier.get_bestfit_Fourier(x,y,yerr,period,Nterms,return_yfit=True,return_params=False,
                                debug=True)
        
        np.testing.assert_allclose(y_fit - y, np.zeros(len(y)))
        
        
        
