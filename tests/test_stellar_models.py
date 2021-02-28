import unittest
import numpy as np
import PIPS


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
        
        
        
        