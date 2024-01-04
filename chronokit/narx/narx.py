import torch
import torch.nn as nn
from chronokit.base._models import Model
from chronokit.preprocessing._dataloader import DataLoader


class NARXModel(torch.nn.Module):
    '''
    Torch implementation of the NARX model.
    '''
    def __init__(self, n_features, hidden_size, batch_norm_momentum=0.0, dropout=0.0):
        super(NARXModel, self).__init__()
        self.Wx = nn.Linear(n_features, hidden_size, bias=False)
        self.Wy = nn.Linear(1, hidden_size)
        if batch_norm_momentum > 0.0:
            self.batch_norm = nn.BatchNorm1d(hidden_size, momentum=batch_norm_momentum)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x, y):
        '''
        Forward pass of the model

        Args:
            x (torch.Tensor): Input tensor (batch_size, n_features)
            y (torch.Tensor): Output tensor (batch_size, 1)

        Returns:
            output (torch.Tensor): Output tensor (batch_size, 1)
        
        '''
        x_out = self.Wx(x)
        y_out = self.Wy(y)
        output = self.tanh(x_out + y_out)
        if hasattr(self, 'batch_norm'):
            output = self.batch_norm(output)
        output = self.dropout(output)
        output = self.output_layer(output)
        return output

class NARX(Model):
    '''
    Nonlinear AutoRegressive with eXogenous inputs (NARX) main model class.
    '''
    def __init__(self,x,y,config={}):        
        # Set allowed kwargs
        self.set_allowed_kwargs(['config'])

        # Initialize config
        self._init_config_(config)
        self.x = DataLoader(x).to_tensor()
        self.y = DataLoader(y).to_tensor()
        self.model = NARXModel(self.config['n_features'], self.config['hidden_size'], self.config['batch_norm_momentum'], self.config['dropout'])

    def _init_config_(self, config):
        '''
        Set default config and update with user provided config

        Args:
            config (dict): User provided config

        Returns:
            None
        
        '''
        # Set default config
        self.config = {
            "batch_size": 10,
            "hidden_size": 10,
            "device": "cpu",
            "n_features": 1,
            "window":1,
            "optimizer": torch.optim.AdamW,
            "lr": 0.001,
            "criterion": torch.nn.MSELoss(),
            "batch_norm_momentum": 0.0, # If 0.0, batch norm is not used
            "dropout": 0.0, # If 0.0, dropout is not used
        }

        # Update config with user provided config
        self.config.update(config)

        # Check if config is valid
        self._check_config()

    def _check_config(self):
        '''
        Checks if the config is valid
        
        Args:
            None

        Returns:
            None

        '''
        # Check if config is valid
        assert self.config['batch_size'] > 0, "batch_size must be greater than 0"
        assert self.config['hidden_size'] > 0, "hidden_size must be greater than 0"
        assert self.config['n_features'] > 0, "n_features must be greater than 0"
        assert self.config['window'] > 0, "window must be greater than 0"
        assert self.config['lr'] > 0, "lr must be greater than 0"
        assert self.config['optimizer'] in [getattr(torch.optim, o) for o in dir(torch.optim) if not o.startswith("_")], "optimizer must be part of torch.optim"
        assert self.config['criterion'] is not None, "criterion cannot be None"
        assert self.config['batch_norm_momentum'] >= 0 and self.config['batch_norm_momentum'] <= 1, "batch_norm_momentum must be between 0 and 1"
        assert self.config['dropout'] >= 0 and self.config['dropout'] <= 1, "dropout must be between 0 and 1"

    def _check_dimensions(self, x, y):
        '''
        Checks if the dimensions of x and y are valid for the model

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Output tensor

        Returns:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Output tensor
        
        '''

        # If 1D, convert to 2D
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)

        return x, y

    def _load_data(self):
        '''
        Checks if the dimensions of x and y are valid for the model and creates a dataloader

        Args:
            None

        Returns:
            dataloader (torch.utils.data.DataLoader): Dataloader
        
        '''
        # Checking dimensions
        self.x, self.y = self._check_dimensions(self.x, self.y)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(self.x, self.y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config['batch_size'],shuffle=False,drop_last=True)
        return dataloader

    def fit(self, epochs=20):
        '''
        Trains the model

        Args:
            epochs (int): Number of epochs to train the model

        Returns:
            train_hist (list): List of training losses
        
        
        '''
        # Loading data and model to device
        dataloader = self._load_data()
        self.model = self.model.to(self.config['device'])

        # Training
        train_hist = []
        criterion = self.config['criterion']
        optimizer = self.config['optimizer'](self.model.parameters(), self.config['lr'])
        total_loss = 0.0

        for epoch in range(epochs):
            self.model.train()
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.config['device']), batch_y.to(self.config['device'])
                optimizer.zero_grad()
                predictions = self.model(batch_X,batch_y)
                loss = criterion(predictions, batch_y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Calculate average training loss and accuracy
            average_loss = total_loss / len(dataloader)
            train_hist.append(average_loss)
            print(f'Epoch [{epoch+1}/{epochs}] - Training Loss: {average_loss:.4f}')

        return train_hist

    def predict(self, x_input, y_input):
        '''
        Predicts the output for a given input

        Args:
            x_input (torch.Tensor): Input tensor (prediction_batch_size, n_features)
            y_input (torch.Tensor): Output tensor

        Returns:
            pred (torch.Tensor): Prediction tensor (prediction_batch_size, 1)
        
        '''
        # Convert to tensor if not already
        if not isinstance(x_input, torch.Tensor) and not isinstance(y_input, torch.Tensor):
            x_input = torch.tensor(x_input).float().to(self.config['device'])
            y_input = torch.tensor(y_input).float().to(self.config['device'])

        # Add an extra dimension at the beginning
        x_input = x_input.unsqueeze(0)
        y_input = y_input.unsqueeze(0)
        self.model.eval()
        pred = self.model(x_input, y_input)
        return pred
        