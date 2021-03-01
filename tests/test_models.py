import unittest
import numpy as np
from PIPS import photdata
import PIPS
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

class TestFourier(unittest.TestCase):
    
    def test_zero_terms(self):
        """
        Should just be a flat line.
        """
        params = np.array([2,1,2, 2]) # arbitrary params
        x = np.linspace(0, 5, 400) # larger than 1 period by far
        Nterms = 0
        period = 1 # arbitrary
        y = PIPS.periodogram.models.Fourier.fourier(x, period, Nterms, params) #
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
        
        
        
