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
        
        
    def test_undo_cut_yerr_max(self):
        object = photdata(self.data)
        object.cut(yerr_max=8)
        object.reset_cuts()
        np.testing.assert_array_equal([object.x, object.y, object.yerr], self.data)
        
    def test_undo_cut_xmin(self):
        object = photdata(self.data)
        object.cut(xmin=2)
        object.reset_cuts()
        np.testing.assert_array_equal([object.x, object.y, object.yerr], self.data)
    
    def test_undo_cut_x_max(self):
        object = photdata(self.data)
        object.cut(xmax=2)
        object.reset_cuts()
        np.testing.assert_array_equal([object.x, object.y, object.yerr], self.data)
        
    def test_undo_cut_ymin(self):
        object = photdata(self.data)
        object.cut(ymin=5)
        object.reset_cuts()
        np.testing.assert_array_equal([object.x, object.y, object.yerr], self.data)
        
    def test_undo_cut_ymax(self):
        object = photdata(self.data)
        object.cut(ymax=5)
        object.reset_cuts()
        np.testing.assert_array_equal([object.x, object.y, object.yerr], self.data)
        
    def test_undo_cut_yerrmin(self):
        object = photdata(self.data)
        object.cut(yerr_min=8)
        object.reset_cuts()
        np.testing.assert_array_equal([object.x, object.y, object.yerr], self.data)
        
    # try second cuts
    def test_second_cut_xmin(self):
        object = photdata(self.data)
        object.cut(xmin=2)
        object.cut(xmin=3)
        self.assertTrue(np.all(object.x >= 3))
    
    def test_second_cut_x_max(self):
        object = photdata(self.data)
        object.cut(xmax=2)
        object.cut(xmax=1)
        self.assertTrue(np.all(object.x <= 1))
        
    def test_second_cut_ymin(self):
        object = photdata(self.data)
        object.cut(ymin=5)
        object.cut(ymin=6)
        self.assertTrue(np.all(object.y >= 6))
        
    def test_second_cut_ymax(self):
        object = photdata(self.data)
        object.cut(ymax=5)
        object.cut(ymax=4)
        self.assertTrue(np.all(object.y <= 4))
        
    def test_second_cut_yerrmin(self):
        object = photdata(self.data)
        object.cut(yerr_min=8)
        object.cut(yerr_min=9)
        self.assertTrue(np.all(object.yerr >= 9))
        
    def test_second_cut_yerrmax(self):
        object = photdata(self.data)
        object.cut(yerr_max=8)
        object.cut(yerr_max=7)
        self.assertTrue(np.all(object.yerr <= 7))
                        
    
                        
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
        output_per, output_err = star.get_period(debug=True)
        self.assertTrue(np.isclose(output_per, expected_per) and np.isclose(output_err, expected_err))
        
    def test_gaussian_fourier_convergence(self):
        """
        The Gaussian and Fourier models should produce similar answers
        for simple data.
        """
        x = np.linspace(0, 100, 1000)
        y = np.sin(x/2)
        yerr = np.ones_like(y) * .01

        star = photdata([x, y, yerr])
        
        gauss_period, guass_err =  star.get_period(
                                                    model='Gaussian', 
                                                    N_peak_test=1000, p_min=0.1,p_max=20)
        fourier_period, fourier_err = star.get_period(model='Fourier', N_peak_test=1000, p_min=0.1,p_max=20)
        
        self.assertTrue(np.isclose(gauss_period, fourier_period))
                        

class TestAmplitudeSpectrum(unittest.TestCase):
    x = np.linspace(0, 100, 1000)
    y = np.sin(x/2)
    yerr = np.ones_like(y) * .01
    
    def test_correct_period(self):
        """
        Test that the correct period is recovered in the amplitude spectrum.
        """
        star = PIPS.photdata([self.x, self.y, self.yerr])

        period,spectrum = star.amplitude_spectrum(p_min=0.1, p_max=20, N=1,multiprocessing=False)
        self.assertTrue(np.isclose(4 * np.pi, period[np.argmax(spectrum)], atol=.001))
        
    def test_single_amplitude(self):
        """
        Test that, for a simple sine function, only a single amplitude is returned.
        """
        star = PIPS.photdata([self.x, self.y, self.yerr])

        period,spectrum = star.amplitude_spectrum(p_min=0.1, p_max=20, N=1,multiprocessing=False)
        self.assertTrue(np.all(spectrum[spectrum!=np.max(spectrum)] == 0))
        
    def test_correct_amplitude(self):
        """
        Test that, for a simple sine function, the correct amplitude is returned.
        """
        star = PIPS.photdata([self.x, self.y, self.yerr])

        period,spectrum = star.amplitude_spectrum(p_min=0.1, p_max=20, N=1,multiprocessing=False)
        self.assertTrue(np.isclose(np.max(spectrum), 2))
    
            
        
