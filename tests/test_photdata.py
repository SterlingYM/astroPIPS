import unittest
import numpy as np
from PIPS import photdata
import PIPS

class TestPhotdataUnit(unittest.TestCase):
    data = np.array([[1,2,3], [4,5,6], [7,8,9]])

    def test_photdata_initialization(self):
        try:
            object = photdata(self.data)
            instantiated = True
        except Exception as e:
            print(e)
            instantiated = False
        self.assertTrue(instantiated)
        
    def test_prepare_data_pass_nones(self):
        object = photdata(self.data)
        try:
            x, y, yerr = object.prepare_data(None, None, None)
            prepared = True
        except Exception as e:
            print(e)
            prepared = False
        self.assertTrue(prepared)
        
        
    def test_prepare_data_pass_vals(self):
        object = photdata(self.data)
        try:
            x, y, yerr = object.prepare_data(
                                            self.data[0],
                                            self.data[1],
                                            self.data[2])
            prepared = True
        except Exception as e:
            print(e)
            prepared = False
        self.assertTrue(prepared)
        
        
    def test_prepare_data_incomplete(self):
        object = photdata(self.data)
        try:
            x, y, yerr = object.prepare_data(
                                            self.data[0],
                                            self.data[1],
                                            None)
            prepared = True
        except Exception as e:
            print(e)
            prepared = False
        self.assertFalse(prepared)
        
    def test_cut_xmin(self):
        object = photdata(self.data)
        object.cut(xmin=2)
        self.assertTrue(np.all(object.x >= 2))
    
    def test_cut_x_max(self):
        object = photdata(self.data)
        object.cut(xmax=2)
        self.assertTrue(np.all(object.x <= 2))
        
    def test_cut_ymin(self):
        object = photdata(self.data)
        object.cut(ymin=5)
        self.assertTrue(np.all(object.y >= 5))
        
    def test_cut_ymax(self):
        object = photdata(self.data)
        object.cut(ymax=5)
        self.assertTrue(np.all(object.y <= 5))
        
    def test_cut_yerrmin(self):
        object = photdata(self.data)
        object.cut(yerr_min=8)
        self.assertTrue(np.all(object.yerr >= 8))
        
    def test_cut_yerrmax(self):
        object = photdata(self.data)
        object.cut(yerr_max=8)
        self.assertTrue(np.all(object.yerr <= 8))
        
                        
    
                        
class TestPhotdataIntegration(unittest.TestCase):
    
    def test_simple_sine_periodogram(self):
        x = np.linspace(0, 100, 1000)
        y = np.sin(x)
        yerr = np.ones_like(y) * .01

        star = photdata([x, y, yerr])
        periods,power = star.periodogram(p_min=0.1,p_max=10,multiprocessing=False)
        max_power = power.max()
        self.assertTrue(np.all(np.isclose(periods[power==power.max()], 2* np.pi, atol=.001)))
        
    def test_regression_get_period(self):
        """
        This tests against older output from get_period.
        """
        expected_per = 0.6968874975991536
        expected_err = 0.0065881527515392994
        data = PIPS.data_readin_LPP('sample_data/005.dat',filter='V')
        x,y,yerr = data
        star = photdata(data)
        output_per, output_err = star.get_period()
        self.assertTrue(np.isclose(output_per, expected_per) and np.isclose(output_err, expected_err))
        
        
                        
                        
                    
                        
                        
        
    
            
        
