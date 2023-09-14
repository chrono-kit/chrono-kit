import unittest
import numpy as np
from chronokit.arima import AutoCorrelation as acs
from statsmodels.tsa.stattools import acf, pacf

class TestACF(unittest.TestCase):
    def setUp(self):
        # Randomly generated data using numpy.random.randind(100,1000,20)
        self.data = np.array([331,123,259,415,664,625,251,102,785,206,745,649,629,788,809,984,748,218,740,412])

        # Control output for acf from R    
        self.acf_expected_r = np.array([1, 0.18546229, 0.18939518, 0.08468901, -0.03005994,0.12644061,0.01702780,-0.09529522,-0.02632613,-0.29098225])
        # Control output for acf from statsmodels
        self.acf_expected_statsmodels = acf(self.data, nlags=9)

        # Control output for pacf from R
        self.pacf_expected_r = np.array([1,0.18546229,0.16052022,0.02703375,-0.08243959,0.13338476,-0.00603359,-0.14606952,-0.00754313,-0.25038150])
        # Control output for pacf from statsmodels
        self.pacf_expected_statsmodels = pacf(self.data, nlags=9, method='ldb')

        # Tolerance values for tests
        self.acf_tol = 1e-6
        self.pacf_tol = 1e-6

    def test_acf_r(self):
        # Test acf function in autocorrelations.py against control output from R
        autocorrelations = acs(self.data)
        acf = autocorrelations.acf(9)
        np.testing.assert_allclose(acf, self.acf_expected_r, rtol=self.acf_tol)

    def test_acf_statsmodels(self):
        # Test acf function in autocorrelations.py against control output from statsmodels
        autocorrelations = acs(self.data)
        acf = autocorrelations.acf(9)
        np.testing.assert_allclose(acf, self.acf_expected_statsmodels, rtol=self.acf_tol)

    def test_pacf_r(self):
        # Test pacf function in autocorrelations.py against control output from R
        autocorrelations = acs(self.data)
        pacf = autocorrelations.pacf(9)
        np.testing.assert_allclose(pacf, self.pacf_expected_r, rtol=self.pacf_tol)

    def test_pacf_statsmodels(self):
        # Test pacf function in autocorrelations.py against control output from statsmodels
        autocorrelations = acs(self.data)
        pacf = autocorrelations.pacf(9)
        np.testing.assert_allclose(pacf, self.pacf_expected_statsmodels, rtol=self.pacf_tol)

if __name__ == '__main__':
    unittest.main()