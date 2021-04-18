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
    x = np.linspace(0, 100, 1000)
    y = np.sin(x/2)
    yerr = np.ones_like(y) * .01
    star = PIPS.photdata([self.x, self.y, self.yerr])
    star.get_period()
    star.get_epoch_offset(model='Fourier', N_peak_test=1000, p_min=0.1,p_max=20)
    
    def test_capture_correct_stellar_type(self):
        correct_string = """Not a valid input. star_type must be a string of the form 'RRab' or 'RRc' in order to work\n"""
        
        obj  = PIPS.Cacciari2005(self.star)
        print_output = io.StringIO()                  
        sys.stdout = print_output                     
        obj.calc_all_vals('')                                     
        sys.stdout = sys.__stdout__                     

        self.assertEqual(print_output.getvalue(), correct_string)
        
        
    def test_regression_log_T_eff_type_ab(self):
        """
        Tests against values in the PIPS paper.
        """ 
        obj  = PIPS.Cacciari2005(self.star)
        
        obj.BV_0 = 0.378
        obj.Fe_H = -2.161
        
        obj.calc_log_T_eff_type_ab()
        
        np.testing.assert_almost_equal(obj.log_T_eff, 3.793, decimal=3)
        
        
    def test_regression_log_T_eff_type_ab(self):
        """
        Tests against values in the PIPS paper.
        """
        obj  = PIPS.Cacciari2005(self.star)
        
        obj.BV_0 = 0.378
        obj.Fe_H = -2.161
        
        obj.calc_log_T_eff_type_ab()
        
        np.testing.assert_almost_equal(obj.log_T_eff, 3.793, decimal=3)
        
    def test_regression_log_L_ab(self):
        """
        Tests against values in the PIPS paper.
        """
        obj  = PIPS.Cacciari2005(self.star)
        
        obj.M_v = 0.540
        
        obj.calc_log_L_ab()
        
        np.testing.assert_almost_equal(obj.log_L, 1.716, decimal=3)
        
        
    def test_regression_log_T_eff_type_ab(self):
        """
        Tests against values in the PIPS paper.
        """
        obj  = PIPS.Cacciari2005(self.star)
        
        obj.BV_0_err = 2.796e-3
        obj.Fe_H_err = 2.001e-3
        
        obj.calc_error_log_T_eff_type_ab()
        
        np.testing.assert_almost_equal(obj.log_T_eff_err, 1.665e-3, decimal=6)
        
    def test_regression_log_L_c(self):
        """
        Tests against values in the PIPS paper.
        """
        obj  = PIPS.Cacciari2005(self.star)
        
        obj.phi21 = 4.532
        obj.M_v = 1.274
        
        obj.calc_log_L_c()
        
        np.testing.assert_almost_equal(obj.log_L, 1.422, decimal=3)
        
    def test_regression_log_surface_gravity_RRab(self):
        """
        Tests against values in the PIPS paper. check works for RRab. TODO
        """
        obj  = PIPS.Cacciari2005(self.star)
        
        obj.log_M = np.log10(0.629)
        obj.log_L = 1.716
        obj.log_T_eff = 3.793
        
        obj.calc_log_surface_gravity()
        
        np.testing.assert_almost_equal(obj.log_g, 2.648, decimal=3)
        
    def test_regression_log_surface_gravity_RRc(self):
        """
        Tests against values in the PIPS paper. check works for RRc stars.
        TODO: check log mass normalization?
        TODO: check log as in log10?
        """
        obj  = PIPS.Cacciari2005(self.star)
        
        obj.log_M = np.log10(.336)
        obj.log_L = 1.274
        obj.log_T_eff = 3.867
        
        obj.calc_log_surface_gravity()
        
        np.testing.assert_almost_equal(obj.log_g, 2.966, decimal=3)
        
    def test_regression_error_log_surface_gravity_RRc(self):
        """
        Tests against values in the PIPS paper. check works for RRc stars.
        """
        obj  = PIPS.Cacciari2005(self.star)
        
        obj.log_M_err = np.log10(9.153e-3)
        obj.log_L_err = 1.126e-5
        obj.log_T_eff_err = 4.660e-4
        
        obj.calc_log_surface_gravity()
        
        np.testing.assert_almost_equal(obj.log_g, 9.341e-3, decimal=3)
        
    def test_calc_all_vals_assign_attrs(self):
        obj  = PIPS.Cacciari2005(self.star)
        obj.calc_all_vals('RRab')
        
        traits = [obj.Fe_H,
                obj.BV_0,
                obj.log_T_eff,
                obj.M_v,
                obj.log_L,
                obj.log_M,
                obj.Fe_H_err,
                obj.BV_0_err,
                obj.log_T_eff_err,
                obj.M_v_err,
                obj.log_L_err,
                obj.log_M_err,
                 ]
        self.assertTrue(np.all(np.isfinite([trait for trait in traits])))
        
