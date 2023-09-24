import unittest
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose as sm_classical
from chronokit.decomposition import classical_decomposition

passengers = pd.read_csv("../datasets/AirPassengers.csv", index_col=0)
sunspots = pd.read_csv("../datasets/Sunspots.csv", index_col=0)
temp = pd.read_csv("../datasets/monthly_temp.csv", index_col=0)

temp1 = temp[temp.columns[0]]
temp2 = temp[temp.columns[1]]

delta = 0.5

class TestClassical(unittest.TestCase):

    def test_passengers_add(self):

        trend, seasonal, remainder = classical_decomposition(data=passengers.values, seasonal_period=12, method="add")
        
        sm_results = sm_classical(passengers.values.squeeze(-1), period=12, model="additive")

        t_diff = np.nanmean(np.sqrt(np.square(trend-sm_results.trend)))/np.nanstd(trend)
        s_diff = np.nanmean(np.sqrt(np.square(seasonal-sm_results.seasonal)))/np.nanstd(seasonal)
        r_diff = np.nanmean(np.sqrt(np.square(remainder-sm_results.resid)))/np.nanstd(remainder)
        
        
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")
    
    def test_temp1(self):

        trend, seasonal, remainder = classical_decomposition(data=temp1.values, seasonal_period=12, method="add")
        
        sm_results = sm_classical(temp1.values, period=12, model="additive")

        t_diff = np.nanmean(np.sqrt(np.square(trend-sm_results.trend)))/np.nanstd(trend)
        s_diff = np.nanmean(np.sqrt(np.square(seasonal-sm_results.seasonal)))/np.nanstd(seasonal)
        r_diff = np.nanmean(np.sqrt(np.square(remainder-sm_results.resid)))/np.nanstd(remainder)
        
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")
    
    def test_temp2(self):

        trend, seasonal, remainder = classical_decomposition(data=temp2.values, seasonal_period=12, method="add")
        
        sm_results = sm_classical(temp2.values, period=12, model="additive")

        t_diff = np.nanmean(np.sqrt(np.square(trend-sm_results.trend)))/np.nanstd(trend)
        s_diff = np.nanmean(np.sqrt(np.square(seasonal-sm_results.seasonal)))/np.nanstd(seasonal)
        r_diff = np.nanmean(np.sqrt(np.square(remainder-sm_results.resid)))/np.nanstd(remainder)
        
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")

    def test_sunspots(self):

        trend, seasonal, remainder = classical_decomposition(data=sunspots.values, seasonal_period=12, method="add")
        
        sm_results = sm_classical(sunspots.values.squeeze(-1), period=12, model="additive")

        t_diff = np.nanmean(np.sqrt(np.square(trend-sm_results.trend)))/np.nanstd(trend)
        s_diff = np.nanmean(np.sqrt(np.square(seasonal-sm_results.seasonal)))/np.nanstd(seasonal)
        r_diff = np.nanmean(np.sqrt(np.square(remainder-sm_results.resid)))/np.nanstd(remainder)
        
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")

if __name__ == "__main__":
    unittest.main()

