import unittest
import pandas as pd
import numpy as np
import random
import chronokit.preprocessing as preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import boxcox

np.random.seed(7)
random.seed(7)

df = pd.read_csv("/home/hasan/Desktop/Codes/yzt_main/datasets/AirPassengers.csv", index_col=0)

arr1 = abs(np.random.randn(100)) + 1
arr2 = abs(np.random.randn(2,100)) +1

delta = 1e-2

class TestTransforms(unittest.TestCase):

    def test_boxcox_df(self):

        BC = preprocessing.BoxCox()
        transformed = BC.transform(df)
        expected = boxcox(df.values.squeeze(1))[0]
        np.testing.assert_allclose(transformed.squeeze(1), expected, delta)

        inv_transformed = BC.inverse_transform(transformed)
        np.testing.assert_allclose(inv_transformed, df.values, delta)

        inv_transformed2 = BC.inverse_transform(transformed, names=list(df.columns))
        np.testing.assert_allclose(inv_transformed2, df.values, delta)
    
    def test_boxcox_arr1(self):
        BC = preprocessing.BoxCox()
        transformed = BC.transform(arr1)
        expected = boxcox(arr1)[0]
        np.testing.assert_allclose(transformed, expected, delta)

        inv_transformed = BC.inverse_transform(transformed)
        np.testing.assert_allclose(inv_transformed, arr1, delta)

        inv_transformed2 = BC.inverse_transform(transformed, names=["data"])
        np.testing.assert_allclose(inv_transformed2, arr1, delta)
    
    def test_boxcox_arr2(self):
        BC = preprocessing.BoxCox()
        transformed = BC.transform(arr2.T)
        expected1 = boxcox(arr2[0,:])[0]
        expected2 = boxcox(arr2[1,:])[0]
        expected = np.zeros((2,100))
        expected[0,:] = expected1
        expected[1,:] = expected2
        np.testing.assert_allclose(transformed.T, expected, delta)

        inv_transformed = BC.inverse_transform(transformed, names=["col0","col1"])
        np.testing.assert_allclose(inv_transformed, arr2.T, delta)

        BC = preprocessing.BoxCox()
        transformed2 = BC.transform(arr2, axis=1)
        np.testing.assert_allclose(transformed2, expected, delta)

        inv_transformed = BC.inverse_transform(transformed2, names=["row0","row1"])
        np.testing.assert_allclose(inv_transformed, arr2, delta)
    
    def test_minmax_df(self):

        MM = preprocessing.MinMaxScaling()
        transformed = MM.transform(df)
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(df)
        np.testing.assert_allclose(transformed, expected, delta)

        inv_transformed = MM.inverse_transform(transformed)
        np.testing.assert_allclose(inv_transformed, df.values, delta)

        inv_transformed2 = MM.inverse_transform(transformed, names=list(df.columns))
        np.testing.assert_allclose(inv_transformed2, df.values, delta)

        MM = preprocessing.MinMaxScaling(feature_range=(-1,1))
        transformed = MM.transform(df)
        scaler = MinMaxScaler(feature_range=(-1,1))
        expected = scaler.fit_transform(df)
        np.testing.assert_allclose(transformed, expected, delta)

        inv_transformed = MM.inverse_transform(transformed)
        np.testing.assert_allclose(inv_transformed, df.values, delta)

        inv_transformed2 = MM.inverse_transform(transformed, names=list(df.columns))
        np.testing.assert_allclose(inv_transformed2, df.values, delta)
    
    def test_minmax_arr1(self):
        MM = preprocessing.MinMaxScaling()
        transformed = MM.transform(arr1)
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(np.expand_dims(arr1, axis=1))[:, 0]
        np.testing.assert_allclose(transformed, expected, delta)

        inv_transformed = MM.inverse_transform(transformed)
        np.testing.assert_allclose(inv_transformed, arr1, delta)

        inv_transformed2 = MM.inverse_transform(transformed, names=["data"])
        np.testing.assert_allclose(inv_transformed2, arr1, delta)
    
    def test_minmax_arr2(self):
        MM = preprocessing.MinMaxScaling()
        transformed = MM.transform(arr2.T)
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(arr2.T)
        np.testing.assert_allclose(transformed, expected, delta)

        inv_transformed = MM.inverse_transform(transformed, names=["row0","row1"])
        np.testing.assert_allclose(inv_transformed, arr2.T, delta)

        MM = preprocessing.MinMaxScaling()
        transformed2 = MM.transform(arr2, axis=1)
        np.testing.assert_allclose(transformed2.T, expected, delta)

        inv_transformed = MM.inverse_transform(transformed2, names=["col0","col1"])
        np.testing.assert_allclose(inv_transformed, arr2, delta)
    
    def test_standard_df(self):

        SS = preprocessing.StandardScaling()
        transformed = SS.transform(df)
        scaler = StandardScaler()
        expected = scaler.fit_transform(df)
        np.testing.assert_allclose(transformed, expected, delta)

        inv_transformed = SS.inverse_transform(transformed)
        np.testing.assert_allclose(inv_transformed, df.values, delta)

        inv_transformed2 = SS.inverse_transform(transformed, names=list(df.columns))
        np.testing.assert_allclose(inv_transformed2, df.values, delta)

    def test_standard_arr1(self):
        SS = preprocessing.StandardScaling()
        transformed = SS.transform(arr1)
        scaler = StandardScaler()
        expected = scaler.fit_transform(np.expand_dims(arr1, axis=1))[:, 0]
        np.testing.assert_allclose(transformed, expected, delta)

        inv_transformed = SS.inverse_transform(transformed)
        np.testing.assert_allclose(inv_transformed, arr1, delta)

        inv_transformed2 = SS.inverse_transform(transformed, names=["data"])
        np.testing.assert_allclose(inv_transformed2, arr1, delta)
    
    def test_standard_arr2(self):
        SS = preprocessing.StandardScaling()
        transformed = SS.transform(arr2.T)
        scaler = StandardScaler()
        expected = scaler.fit_transform(arr2.T)
        np.testing.assert_allclose(transformed, expected, delta)

        inv_transformed = SS.inverse_transform(transformed, names=["row0","row1"])
        np.testing.assert_allclose(inv_transformed, arr2.T, delta)

        SS = preprocessing.StandardScaling()
        transformed2 = SS.transform(arr2, axis=1)
        np.testing.assert_allclose(transformed2.T, expected, delta)

        inv_transformed = SS.inverse_transform(transformed2, names=["col0","col1"])
        np.testing.assert_allclose(inv_transformed, arr2, delta)

        


if __name__ == "__main__":
    unittest.main()