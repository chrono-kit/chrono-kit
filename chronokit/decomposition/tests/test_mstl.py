import unittest
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import MSTL as sm_mstl
from chronokit.decomposition import MSTL
from chronokit.utils import metrics
from chronokit.preprocessing.data_transforms import MinMaxScaling

# (1,2) for testing multiplicative decomposition
scaler = MinMaxScaling(feature_range=(1,2))

passengers = pd.read_csv("../datasets/AirPassengers.csv", index_col=0)
sunspots = pd.read_csv("../datasets/Sunspots.csv", index_col=0)
temp = pd.read_csv("../datasets/monthly_temp.csv", index_col=0)

passengers = scaler.transform(passengers.values).squeeze(-1)
sunspots = scaler.transform(sunspots.values).squeeze(-1)

temp1 = scaler.transform(temp[temp.columns[0]].values)
temp2 = scaler.transform(temp[temp.columns[1]].values)

delta = 0.1

class TestMSTL(unittest.TestCase):

    def test_passengers(self):

        trend, seasonal, remainder = MSTL(data=passengers, seasonal_periods=[4,12],
                                          robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_mstl(passengers, periods=[4,12], stl_kwargs={"outer_iter":10, "inner_iter":2, "robust": True}).fit()
        t_diff = metrics.rmse(trend, sm_results.trend)
        s_diff = metrics.rmse(seasonal, sm_results.seasonal.T)
        r_diff = metrics.rmse(remainder, sm_results.resid)
        
        print(t_diff, s_diff, r_diff)
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")
    
    def test_passengers_mul(self):

        trend, seasonal, remainder = MSTL(data=passengers, seasonal_periods=[4,12], method="mul",
                                          robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_mstl(np.log(passengers), periods=[4,12], stl_kwargs={"outer_iter":10, "inner_iter":2, "robust": True}).fit()

        t_diff = metrics.rmse(trend, np.exp(sm_results.trend))
        s_diff = metrics.rmse(seasonal, np.exp(sm_results.seasonal).T)
        r_diff = metrics.rmse(remainder, np.exp(sm_results.resid))
        
        print(t_diff, s_diff, r_diff)
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")
    
    def test_temp1(self):

        trend, seasonal, remainder = MSTL(data=temp1, seasonal_periods=[4,12],
                                          robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_mstl(temp1, periods=[4,12], stl_kwargs={"outer_iter":10, "inner_iter":2, "robust": True}).fit()

        t_diff = metrics.rmse(trend, sm_results.trend)
        s_diff = metrics.rmse(seasonal, sm_results.seasonal.T)
        r_diff = metrics.rmse(remainder, sm_results.resid)
        
        print(t_diff, s_diff, r_diff)
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")
    
    def test_temp1_mul(self):

        trend, seasonal, remainder = MSTL(data=temp1, seasonal_periods=[4,12], method="mul",
                                          robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_mstl(np.log(temp1), periods=[4,12], stl_kwargs={"outer_iter":10, "inner_iter":2, "robust": True}).fit()

        t_diff = metrics.rmse(trend, np.exp(sm_results.trend))
        s_diff = metrics.rmse(seasonal, np.exp(sm_results.seasonal).T)
        r_diff = metrics.rmse(remainder, np.exp(sm_results.resid))
        
        print(t_diff, s_diff, r_diff)
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")
    
    def test_temp2(self):

        trend, seasonal, remainder = MSTL(data=temp2, seasonal_periods=[4,12],
                                          robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_mstl(temp2, periods=[4,12], stl_kwargs={"outer_iter":10, "inner_iter":2, "robust": True}).fit()

        t_diff = metrics.rmse(trend, sm_results.trend)
        s_diff = metrics.rmse(seasonal, sm_results.seasonal.T)
        r_diff = metrics.rmse(remainder, sm_results.resid)
        
        print(t_diff, s_diff, r_diff)
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")
    
    def test_temp2_mul(self):

        trend, seasonal, remainder = MSTL(data=temp2, seasonal_periods=[4,12], method="mul",
                                          robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_mstl(np.log(temp2), periods=[4,12], stl_kwargs={"outer_iter":10, "inner_iter":2, "robust": True}).fit()

        t_diff = metrics.rmse(trend, np.exp(sm_results.trend))
        s_diff = metrics.rmse(seasonal, np.exp(sm_results.seasonal).T)
        r_diff = metrics.rmse(remainder, np.exp(sm_results.resid))
        
        print(t_diff, s_diff, r_diff)
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")
    
    def test_sunspots(self):

        trend, seasonal, remainder = MSTL(data=sunspots, seasonal_periods=[4,12],
                                          robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_mstl(sunspots, periods=[4,12], stl_kwargs={"outer_iter":10, "inner_iter":2, "robust": True}).fit()

        t_diff = metrics.rmse(trend, sm_results.trend)
        s_diff = metrics.rmse(seasonal, sm_results.seasonal.T)
        r_diff = metrics.rmse(remainder, sm_results.resid)
        
        print(t_diff, s_diff, r_diff)
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")
    
    def test_sunspots_mul(self):

        trend, seasonal, remainder = MSTL(data=sunspots, seasonal_periods=[4,12], method="mul",
                                          robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_mstl(np.log(sunspots), periods=[4,12], stl_kwargs={"outer_iter":10, "inner_iter":2, "robust": True}).fit()

        t_diff = metrics.rmse(trend, np.exp(sm_results.trend))
        s_diff = metrics.rmse(seasonal, np.exp(sm_results.seasonal).T)
        r_diff = metrics.rmse(remainder, np.exp(sm_results.resid))
        
        print(t_diff, s_diff, r_diff)
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")

if __name__ == "__main__":
    unittest.main()

