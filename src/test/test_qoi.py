import unittest
import numpy as np

import sys
sys.path.append("../")
from dt4co.qoi import compute_ccc

class TestCCC(unittest.TestCase):
    def test_ccc(self):
        # test the compute_ccc function
        # Test case is from: https://rowannicholls.github.io/python/statistics/agreement/concordance_correlation_coefficient.html
        # which uses: https://github.com/stylianos-kampakis/supervisedPCA-Python/blob/master/Untitled.py
        
        x = np.array([2.5, 0.0, 2, 8])
        y = np.array([3, -0.5, 2, 7])
        
        ccc = compute_ccc(x, y, bias=False, use_pearson=True)
        
        VAL = 0.97678916827853024  # expected value
        
        assert np.isclose(ccc, VAL, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
