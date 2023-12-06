import unittest
import numpy as np
import numpy.testing as npt
from chronokit.narx import NARX
import torch

class TestNARX(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)

        # Testing two dimensional input and single dimensional output
        self.x1 = np.random.rand(100,2)
        self.y1 = np.random.rand(100,1)
        self.n1 = 2
        self.model1 = NARX(self.x1,self.y1,n_features=self.n1,batch_size=10,hidden_size=10,device='cpu')

        # Testing three dimensional input and single dimensional output
        self.x2 = np.random.rand(100,3)
        self.y2 = np.random.rand(100,1)
        self.n2 = 3
        self.model2 = NARX(self.x2,self.y2,n_features=self.n2,batch_size=10,hidden_size=10,device='cpu')

        # Testing single dimensional input and single dimensional output
        self.x3 = np.random.rand(100,1)
        self.y3 = np.random.rand(100,1)
        self.n3 = 1
        self.model3 = NARX(self.x3,self.y3,n_features=self.n3,batch_size=10,hidden_size=10,device='cpu')

        # Testing single dimensional input and two dimensional output
        # Should fail
        self.x4 = np.random.rand(100,1)
        self.y4 = np.random.rand(100,2)
        self.n4 = 1
        self.model4 = NARX(self.x4,self.y4,n_features=self.n4,batch_size=10,hidden_size=10,device='cpu')

        # Testing single dimensional input and single dimensional output
        self.x5 = np.random.rand(100,1)
        self.n5 = 1
        self.model5 = NARX(self.x5,self.x5,n_features=self.n5,batch_size=10,hidden_size=10,device='cpu')


    def test_init(self):
        # Testing initialization of NARX model
        self.assertIsInstance(self.model1, NARX)
        self.assertIsInstance(self.model2, NARX)
        self.assertIsInstance(self.model3, NARX)
        self.assertIsInstance(self.model4, NARX)
        self.assertIsInstance(self.model5, NARX)


    def test_check_dimensions(self):
        # Testing _check_dimensions method
        x1, y1 = self.model1._check_dimensions(self.x1, self.y1)
        x2, y2 = self.model2._check_dimensions(self.x2, self.y2)
        x3, y3 = self.model3._check_dimensions(self.x3, self.y3)
        x4, y4 = self.model4._check_dimensions(self.x4, self.y4)
        x5, y5 = self.model5._check_dimensions(self.x5, self.x5)

        # Testing dimensions of output
        self.assertEqual(x1.shape, (100,2))
        self.assertEqual(y1.shape, (100,1))
        self.assertEqual(x2.shape, (100,3))
        self.assertEqual(y2.shape, (100,1))
        self.assertEqual(x3.shape, (100,1))
        self.assertEqual(y3.shape, (100,1))
        self.assertEqual(x4.shape, (100,1))
        self.assertEqual(y4.shape, (100,2))
        self.assertEqual(x5.shape, (100,1))
        self.assertEqual(y5.shape, (100,1))


    def test_load_data(self):
        # Testing _load_data method
        dataloader1 = self.model1._load_data()
        dataloader2 = self.model2._load_data()
        dataloader3 = self.model3._load_data()
        dataloader4 = self.model4._load_data()
        dataloader5 = self.model5._load_data()

        # Testing dataloader
        self.assertEqual(len(dataloader1), 10)
        self.assertEqual(len(dataloader2), 10)
        self.assertEqual(len(dataloader3), 10)
        self.assertEqual(len(dataloader4), 10)
        self.assertEqual(len(dataloader5), 10)

    def test_fit(self):
        # Testing fit method
        self.model1.fit(epochs=20, lr=0.001)
        self.model2.fit(epochs=20, lr=0.001)
        self.model3.fit(epochs=20, lr=0.001)
        # Model4 should fail because of dimension mismatch
        self.assertRaises(RuntimeError, self.model4.fit, epochs=20, lr=0.001)
        self.model5.fit(epochs=20, lr=0.001)

        # Testing prediction
        y_pred1 = self.model1.predict(self.x1[-1], self.y1[-1])
        y_pred2 = self.model2.predict(self.x2[-1], self.y2[-1])
        y_pred3 = self.model3.predict(self.x3[-1], self.y3[-1])
        y_pred5 = self.model5.predict(self.x5[-2], self.x5[-1])

        # Testing dimensions of output
        self.assertEqual(y_pred1.shape, torch.Size([1]))
        self.assertEqual(y_pred2.shape, torch.Size([1]))
        self.assertEqual(y_pred3.shape, torch.Size([1]))


if __name__ == '__main__':
    unittest.main()




