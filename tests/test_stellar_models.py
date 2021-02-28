import unittest
import numpy as np
import PIPS


class TestStellarInit(unittest.TestCase):
    
    def test_init_not_photdata(self):
        not_photdata = [] # empty list that isn't a photdata object
        obj = PIPS.StellarModels(not_photdata)
        try:
            obj.C_k
            correctly_instantiated = False
        except AttributeError:
            # the object shouldn't have instantiated because it's not a photdata object!
            correctly_instantiated = True 
            
        self.assertTrue(correctly_instantiated)