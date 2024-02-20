import torch
import unittest
import numpy as np
import numpy.testing as npt
from chronokit.networks import NARX

class TestNARX(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)

        # Testing two dimensional input and single dimensional output
        self.x1 = np.random.rand(100,2)
        self.y1 = np.random.rand(100,1)
        self.config1 = {
            "batch_size": 10,
            "hidden_size": 10,
            "device": "cpu",
            "n_features": 2,
            "window":1,
            "optimizer": torch.optim.AdamW,
            "lr": 0.001, 
            "criterion": torch.nn.MSELoss(),
            "batch_norm_momentum": 0.1,
            "dropout": 0.2,
        }
        self.model1 = NARX(self.x1,self.y1,self.config1)

        # Testing three dimensional input and single dimensional output
        self.x2 = np.random.rand(100,3)
        self.y2 = np.random.rand(100,1)
        self.config2 = {
            "n_features": 3,
            "optimizer": torch.optim.AdamW,
            "lr": 0.001, 
            "criterion": torch.nn.MSELoss(),
            "batch_norm_momentum": 0.1,
            "dropout": 0.2,
        }
        self.model2 = NARX(self.x2,self.y2,self.config2)

        # Testing single dimensional input and single dimensional output
        self.x3 = np.random.rand(100,1)
        self.y3 = np.random.rand(100,1)

        self.model3 = NARX(self.x3,self.y3)

        # Testing single dimensional input and two dimensional output
        # Should fail
        self.x4 = np.random.rand(100,1)
        self.y4 = np.random.rand(100,2)
        self.model4 = NARX(self.x4,self.y4)

        # Testing single dimensional input and single dimensional output
        self.x5 = np.random.rand(100,1)
        self.model5 = NARX(self.x5,self.x5)

        # Testing different optimizer
        self.x67 = np.random.rand(100,1)
        self.y67 = np.random.rand(100,1)
        self.config6 = {
            "optimizer": torch.optim.SGD,
        }
        self.model6 = NARX(self.x67,self.y67,self.config6)

        # Testing different criterion
        self.config7 = {
            "criterion": torch.nn.L1Loss(),

        }
        self.model7 = NARX(self.x67,self.y67,self.config7)



    def test_init(self):
        # Testing initialization of NARX model
        self.assertIsInstance(self.model1, NARX)
        self.assertIsInstance(self.model2, NARX)
        self.assertIsInstance(self.model3, NARX)
        self.assertIsInstance(self.model4, NARX)
        self.assertIsInstance(self.model5, NARX)
        self.assertIsInstance(self.model6, NARX)


    def test_check_dimensions(self):
        # Testing _check_dimensions method
        x1, y1 = self.model1._check_dimensions(self.x1, self.y1)
        x2, y2 = self.model2._check_dimensions(self.x2, self.y2)
        x3, y3 = self.model3._check_dimensions(self.x3, self.y3)
        x4, y4 = self.model4._check_dimensions(self.x4, self.y4)
        x5, y5 = self.model5._check_dimensions(self.x5, self.x5)
        x67, y67 = self.model6._check_dimensions(self.x67, self.y67)

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
        self.assertEqual(x67.shape, (100,1))
        self.assertEqual(y67.shape, (100,1))


    def test_load_data(self):
        # Testing _load_data method
        dataloader1 = self.model1._load_data()
        dataloader2 = self.model2._load_data()
        dataloader3 = self.model3._load_data()
        dataloader4 = self.model4._load_data()
        dataloader5 = self.model5._load_data()
        dataloader6 = self.model6._load_data()
        dataloader7 = self.model7._load_data()

        # Testing dataloader
        self.assertEqual(len(dataloader1), 10)
        self.assertEqual(len(dataloader2), 10)
        self.assertEqual(len(dataloader3), 10)
        self.assertEqual(len(dataloader4), 10)
        self.assertEqual(len(dataloader5), 10)
        self.assertEqual(len(dataloader6), 10)
        self.assertEqual(len(dataloader7), 10)

    def test_fit(self):
        # Testing fit method
        self.model1.fit(epochs=20)
        self.model2.fit(epochs=20)
        self.model3.fit(epochs=20)
        # Model4 should fail because of dimension mismatch
        self.assertRaises(RuntimeError, self.model4.fit, epochs=20)
        self.model5.fit(epochs=20)
        self.model6.fit(epochs=20)
        self.model7.fit(epochs=20)

        # Testing prediction
        y_pred1 = self.model1.predict(self.x1[-1], self.y1[-1])
        y_pred2 = self.model2.predict(self.x2[-1], self.y2[-1])
        y_pred3 = self.model3.predict(self.x3[-1], self.y3[-1])
        y_pred5 = self.model5.predict(self.x5[-2], self.x5[-1])
        y_pred6 = self.model6.predict(self.x67[-1], self.y67[-1])
        y_pred7 = self.model7.predict(self.x67[-1], self.y67[-1])



        # Testing dimensions of output
        self.assertEqual(y_pred1.shape, torch.Size([1,1]))
        self.assertEqual(y_pred2.shape, torch.Size([1,1]))
        self.assertEqual(y_pred3.shape, torch.Size([1,1]))
        self.assertEqual(y_pred5.shape, torch.Size([1,1]))
        self.assertEqual(y_pred6.shape, torch.Size([1,1]))
        self.assertEqual(y_pred7.shape, torch.Size([1,1]))


if __name__ == '__main__':
    unittest.main()
