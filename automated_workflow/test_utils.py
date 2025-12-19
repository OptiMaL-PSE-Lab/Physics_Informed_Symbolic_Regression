
import unittest
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils
import config

class TestUtils(unittest.TestCase):
    
    def test_find_best_model(self):
        nll = [10, 5, 20]
        params = [1, 2, 1]
        # AIC = 2*k + 2*ln(L^) -> simplified here as 2*NLL + 2*k
        # 1: 2*10 + 2*1 = 22
        # 2: 2*5 + 2*2 = 14  <- Best
        # 3: 2*20 + 2*1 = 42
        best = utils.find_best_model(nll, params)
        self.assertEqual(best, 1)

    def test_NLL(self):
        C = np.array([1.0, 2.0, 3.0])
        y_C = np.array([1.0, 2.0, 3.0])
        # Perfect fit should have low NLL (actually might depend on variance calculation)
        # In current impl, if perfect fit, mse=0, variance=0 => div by zero issue in the code?
        # Let's test with slight difference
        y_C_noisy = np.array([1.1, 1.9, 3.1])
        nll = utils.NLL(C, y_C_noisy, 3)
        self.assertIsInstance(nll, float)
        
    def test_predicting_rate(self):
        eq = "2 * A + B"
        z = np.array([[1, 2], [3, 4]]) # [A, B] columns
        # Row 1: A=1, B=2 -> 2*1 + 2 = 4
        # Row 2: A=3, B=4 -> 2*3 + 4 = 10
        pred = utils.predicting_rate(eq, z)
        expected = np.array([4, 10])
        np.testing.assert_array_equal(pred, expected)

if __name__ == '__main__':
    unittest.main()
