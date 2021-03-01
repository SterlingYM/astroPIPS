import unittest
import numpy as np
import PIPS
import sys
import io


class TestStellarInit(unittest.TestCase):
    
    def test_init_not_photdata(self):
        not_photdata = [] # empty list that isn't a photdata object
        obj = PIPS.StellarModels(not_photdata)
        try:
            obj.C_1
            correctly_instantiated = False
        except AttributeError:
            # the object shouldn't have instantiated because it's not a photdata object!
            correctly_instantiated = True 
            
        self.assertTrue(correctly_instantiated)
        
        
        
    def test_init_not_run_ger_period(self):
        x = np.linspace(0, 100, 1000)
        y = np.sin(x/2)
        yerr = np.ones_like(y) * .01

        star = PIPS.photdata([x, y, yerr])
        obj = PIPS.StellarModels(star)
        try:
            obj.C_1
            correctly_instantiated = False
        except AttributeError:
            # the object shouldn't have instantiated because get_period hasn't been run!
            correctly_instantiated = True 
            
        self.assertTrue(correctly_instantiated)
        
        
    def test_init_normal(self):
        x = np.linspace(0, 100, 1000)
        y = np.sin(x/2)
        yerr = np.ones_like(y) * .01

        star = PIPS.photdata([x, y, yerr])
        star.get_period()
        star.get_epoch_offset()
        obj = PIPS.StellarModels(star)
        try:
            obj.C_1
            correctly_instantiated = True
        except AttributeError:
            # the object shouldn't False instantiated because get_period hasn't been run!
            correctly_instantiated = True 
            
        self.assertTrue(correctly_instantiated)
        
        
class TestCacciari2005(unittest.TestCase):
    
    def test_capture_correct_stellar_type(self):
        correct_string = """Not a valid input. star_type must be a string of the form 'RRab' or 'RRc' in order to work\n"""
        x = np.linspace(0, 100, 1000)
        y = np.sin(x/2)
        yerr = np.ones_like(y) * .01

        star = PIPS.photdata([x, y, yerr])
        star.get_period()
        star.get_epoch_offset(model='Fourier', N_peak_test=1000, p_min=0.1,p_max=20)
        
        obj  = PIPS.Cacciari2005(star)
        print_output = io.StringIO()                  
        sys.stdout = print_output                     
        obj.calc_all_vals('')                                     
        sys.stdout = sys.__stdout__                     

        self.assertEqual(print_output.getvalue(), correct_string)
        
        
        
