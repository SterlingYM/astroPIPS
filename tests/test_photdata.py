import unittest
from . import photdata

class TestPhotdata(unittest.TestCase):

    def test_photdata_initialization(self):
        try:
            object = photdata()
            instantiated = True
        except:
            instantiated = False
        self.assertTrue(instantiated)