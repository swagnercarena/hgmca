from hgmca import hgmca_core, wavelets
import numpy as np
import unittest
import os


class HGMCATests(unittest.TestCase):

	def setUp(self, *args, **kwargs):
		self.root_path = os.path.dirname(os.path.abspath(__file__))+'/test_data/'
		# Remember, healpy expects radians but we use arcmins.
		self.a2r = np.pi/180/60
