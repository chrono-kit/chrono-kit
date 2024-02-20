import unittest
import torch
from chronokit.nn import NeuralNetwork

class TestNARX(unittest.TestCase):
    def setUp(self):
        # Set random seed
        torch.manual_seed(0)

        # Set up architectures
        # Architecture 1 should be valid
        self.architecture1 = {
            "layers": {
                "linear_1": torch.nn.Linear(1, 1)
            },
            "activations": {
                "relu_1": torch.nn.ReLU()
            },
            "forward": ["linear_1", "relu_1"]
        }

        # Architecture 2 should be invalid because it does not contain layers
        self.architecture2 = {
            "activations": {
                "relu_1": torch.nn.ReLU()
            },
            "forward": ["linear_1", "relu_1"]
        }

        # Architecture 3 should be invalid because it does not contain activations
        self.architecture3 = {
            "layers": {
                "linear_1": torch.nn.Linear(1, 1)
            },
            "forward": ["linear_1", "relu_1"]
        }

        # Architecture 4 should be invalid because it does not contain forward
        self.architecture4 = {
            "layers": {
                "linear_1": torch.nn.Linear(1, 1)
            },
            "activations": {
                "relu_1": torch.nn.ReLU()
            }
        }

        # Architecture 5 should be invalid because layers is not a dictionary
        self.architecture5 = {
            "layers": torch.nn.Linear(1, 1),
            "activations": {
                "relu_1": torch.nn.ReLU()
            },
            "forward": ["linear_1", "relu_1"]
        }

        # Architecture 6 should be invalid because activations is not a dictionary
        self.architecture6 = {
            "layers": {
                "linear_1": torch.nn.Linear(1, 1)
            },
            "activations": torch.nn.ReLU(),
            "forward": ["linear_1", "relu_1"]
        }

    def test_creating_models(self):
        # Test creating a valid model
        model1 = NeuralNetwork(self.architecture1)
        self.assertTrue(isinstance(model1, NeuralNetwork))

        # Test creating an invalid model
        with self.assertRaises(ValueError):
            model2 = NeuralNetwork(self.architecture2)
        with self.assertRaises(ValueError):
            model3 = NeuralNetwork(self.architecture3)
        with self.assertRaises(ValueError):
            model4 = NeuralNetwork(self.architecture4)
        with self.assertRaises(ValueError):
            model5 = NeuralNetwork(self.architecture5)
        with self.assertRaises(ValueError):
            model6 = NeuralNetwork(self.architecture6)

    def test_forward(self):
        # Test forward pass
        # Should return a tensor of shape [1]
        model1 = NeuralNetwork(self.architecture1)
        x = torch.randn(1)
        y = model1(x)
        self.assertEqual(y.shape, torch.Size([1]))

if __name__ == '__main__':
    unittest.main()
