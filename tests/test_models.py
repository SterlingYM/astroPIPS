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
        
        np.testing.assert_allclose(y_fit, y, atol=1e-10)
        
        
class TestGaussian(unittest.TestCase):
    
    def test_somewhat_close(self):
        """
        Gaussian fit should somewhat fit the sine curve.
        """
        x = np.linspace(0, 100, 1000)
        y = np.sin(x/2)
        yerr = np.ones_like(y) * .01

        period = 4 * np.pi

        Nterms = 1

        y_fit = PIPS.periodogram.models.Gaussian.get_bestfit_gaussian(x,y,yerr,period,Nterms,return_yfit=True,return_params=False,
                                debug=True)
        np.testing.assert_allclose(y, y_fit, atol=.8)
        

class TestPeriodograms(unittest.TestCase):
    def test_periodogram_convergence(self):
        """
        both periodogram methods should roughly line up!
        """
        x = np.linspace(0, 100, 1000)
        y = np.sin(x)
        yerr = np.ones_like(y) * .01

        star = PIPS.photdata([x, y, yerr])
        periods, power_cust = PIPS.periodogram.custom.periodogram_custom(0.1,10,100, x, y, yerr, Nterms=1, multiprocessing=True)
        periods, power_fast = PIPS.periodogram.linalg.periodogram_fast(0.1,10,100, x, y, yerr, Nterms=1, multiprocessing=True)
        np.testing.assert_allclose(power_fast, power_cust, atol=1e-3)
        
        
    def test_periodogram_custom_periods(self):
        """
        both periodogram methods should roughly line up!
        """
        x = np.linspace(0, 100, 1000)
        y = np.sin(x)
        yerr = np.ones_like(y) * .01
        
        custom_periods = np.linspace(0.1, 10, 100)

        star = PIPS.photdata([x, y, yerr])
        periods, power_custom_periods = PIPS.periodogram.custom.periodogram_custom(None,None,100, x, y, yerr, Nterms=1, multiprocessing=True,
                                                                                  custom_periods=custom_periods)
        periods, power = PIPS.periodogram.custom.periodogram_custom(0.1,10,100, x, y, yerr, Nterms=1, multiprocessing=True)
        np.testing.assert_allclose(power, power_custom_periods, atol=1e-11)
        
