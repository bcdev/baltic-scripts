import os
import sys
import unittest
from sys import platform
sys.path.append("C:\\Users\Telpecarne\.snap\snap-python")
import baltic_ac_algorithm

BALTIC_AC_HOME = os.path.dirname(os.path.abspath(__file__))

# noinspection PyUnresolvedReferences
class TestBalticAc(unittest.TestCase):
    def setUp(self):
        print('Platform: ', platform)
        # parent_dir = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))

        resource_root = os.path.dirname(__file__)

        sys.path.append(resource_root)
        sys.path.append(BALTIC_AC_HOME)
        self.baltic_ac_algo = baltic_ac_algorithm.BalticAcAlgorithm()

    def test_balticac(self):
        # todo
        pass


print('Testing balticac')
suite = unittest.TestLoader().loadTestsFromTestCase(TestBalticAc)
unittest.TextTestRunner(verbosity=2).run(suite)
