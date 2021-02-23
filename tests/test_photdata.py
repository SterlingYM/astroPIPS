import unittest
import numpy as np
from PIPS import photdata

class TestPhotdata(unittest.TestCase):
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
        
                        
    
                        
                        
                        
                    
                        
                        
        
    
            
        
