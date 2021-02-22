import unittest
from PIPS import photdata

class TestPhotdata(unittest.TestCase):

    def test_photdata_initialization(self):
        try:
            object = photdata()
            instantiated = True
        except Exception as e:
            print(e)
            instantiated = False
        self.assertTrue(instantiated)
