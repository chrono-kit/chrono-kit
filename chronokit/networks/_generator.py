import torch.nn as nn

class TSNeuralNetwork(nn.Module):
    def __init__(self,architecture):
        '''
        This class takes a dictionary of layers and parameters and creates
        a pytorch model with a forward loop from it.

        Args:
            architecture : dict
                Dictionary containing the architecture of the network.
                Example architecture:
                architecture = {
                    "layers":{
                        "layer1": nn.Linear(10, 5),
                        "layer2": nn.Linear(5, 1)
                    },
                    "activations":{
                        "activation1": nn.ReLU(),
                        "activation2": nn.ReLU()
                    },
                    forward: ["layer1", "activation1", "layer2", "activation2"]
                }
        '''

        raise NotImplementedError(
            "This class is unavailable as of v1.1.x"
        )
        super(NeuralNetwork, self).__init__()
        self.architecture = architecture
        # Check if architecture is valid
        self._check_architecture()

        # Create layers if architecture is valid
        self.layers = nn.ModuleDict(architecture["layers"])

        if "activations" in architecture:
            self.activations = nn.ModuleDict(architecture["activations"])
        else:
            self.activations = None

        self.nn_forward = architecture["forward"]

    def _check_architecture(self):
        """Checks if the architecture is valid"""

        # Check if architecture is a dictionary
        if not isinstance(self.architecture, dict):
            raise ValueError("Neural Network model architecture must be a dictionary")
        
        # Check if architecture contains layers
        if "layers" not in self.architecture:
            raise ValueError("Neural Network model architecture must contain 'layers'")
        # Check if layers are in a dictionary and not empty
        if not isinstance(self.architecture["layers"], dict) or len(self.architecture["layers"]) == 0:
            raise ValueError("Neural Network model architecture 'layers' must be a non empty dictionary")
    
        if "activations" in self.architecture:
        # Check if activations are in a dictionary and not empty
            if not isinstance(self.architecture["activations"], dict) or len(self.architecture["activations"]) == 0:
                raise ValueError("Neural Network model architecture activations must be a non empty dictionary")
        
        # Check if architecture contains forward
        if "forward" not in self.architecture:
            raise ValueError("Neural Network model architecture must contain forward list")
        # Check if forward is a list and not empty
        if not isinstance(self.architecture["forward"], list) or len(self.architecture["forward"]) == 0:
            raise ValueError("Neural Network model architecture forward must be a non empty list")

    def forward(self, x):
        '''
        Forward pass of the neural network

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        '''
        for layer in self.nn_forward:
            if layer in self.layers:
                x = self.layers[layer](x)
            elif layer in self.activations:
                x = self.activations[layer](x)
            else:
                raise ValueError("Invalid layer in forward list")

        return x
