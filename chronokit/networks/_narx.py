import torch
import torch.nn as nn
from chronokit.preprocessing._dataloader import DataLoader
from chronokit.base._models import NeuralTimeSeriesModel


class NARXModel(nn.Module):
    """
    Torch implementation of Univariate NARX model.
    
    Implementation was made with mathworks article [1] as reference

    [1] https://www.mathworks.com/help/deeplearning/ug/design-time-series-narx-feedback-neural-networks.html
    """

    def __init__(
                self, 
                window,
                exog_window,
                hidden_size, 
                batch_norm_momentum=0.0, 
                dropout=0.0,
                device = "cpu"
    ):
        
        super(NARXModel, self).__init__()
        torch.manual_seed(7)

        self.Wy = nn.Parameter(data=torch.randn((window, hidden_size), dtype=torch.float32), requires_grad=True).to(device)
        self.Wx = nn.Parameter(data=torch.randn((exog_window, hidden_size), dtype=torch.float32), requires_grad=True).to(device)
        self.bias = nn.Parameter(data=torch.zeros((1), dtype=torch.float32), requires_grad=True).to(device)

        if batch_norm_momentum > 0.0:
            self.batch_norm = nn.BatchNorm1d(hidden_size, momentum=batch_norm_momentum).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.output_layer = nn.Linear(hidden_size, 1, dtype=torch.float32).to(device)
        self.tanh = nn.Tanh()
        
        #Only Xavier Initialization is available as of v1.1.x
        #More will be available later
        nn.init.xavier_normal_(self.Wx)
        nn.init.xavier_normal_(self.Wx)
        nn.init.xavier_normal_(self.output_layer.weight)

    def forward(
            self,
            endog_inputs, 
            exog_inputs
):
        '''
        Forward pass of the model

        Args:
            endog_inputs (torch.Tensor): Endogenous input tensor (batch_size, window)
            exog_inputs (torch.Tensor): Exogenous input tensor (batch_size, exog_window)

        Returns:
            output (torch.Tensor): Output tensor (batch_size, n_features)
        
        '''
        y_out = torch.matmul(endog_inputs, self.Wy) #shape: (batch, hidden)
        x_out = torch.matmul(exog_inputs, self.Wx) #shape: (batch, hidden)

        grouped = x_out + y_out + self.bias #shape: (batch, hidden)

        output = self.tanh(grouped)
        if hasattr(self, 'batch_norm'):
            output = self.batch_norm(output.transpose(1,2)).transpose(1,2)
        
        output = self.dropout(output)
        output = self.output_layer(output) #shape: (batch, 1)

        return output

class NARX(NeuralTimeSeriesModel):
    def __init__(
                self,
                data,
                exogenous_data,
                config = {}
):       
        """
        Neural Network based NARX model for univariate time series modeling.

        Arguments:

        * data (array_like): Univarite time series data to model
        * exogenous_data (array_like): Exogenous data to use for modeling
            the data. Must be of the same length as the data.
        * config (Optional[dict]): Configurations for the model architecture
            Default: {
                    "batch_size": 10,
                    "hidden_size": 10,
                    "device": "cpu",
                    "window":1, 
                    "exog_window": 1,
                    "optimizer": torch.optim.AdamW, 
                    "lr": 0.001,
                    "criterion": torch.nn.MSELoss(),
                    "batch_norm_momentum": 0.0, # If 0.0, batch norm is not used
                    "dropout": 0.0, # If 0.0, dropout is not used
            }
    
        Implementation was made with mathworks article [1] as reference

        [1] https://www.mathworks.com/help/deeplearning/ug/design-time-series-narx-feedback-neural-networks.html
        """
        super().__init__(data, exogenous_data=exogenous_data)

        # Initialize config
        self._init_config_(config)
        self.model = NARXModel(
                window=self.config["window"],
                exog_window=self.config["exog_window"],
                hidden_size=self.config['hidden_size'], 
                batch_norm_momentum=self.config['batch_norm_momentum'], 
                dropout=self.config['dropout'],
                device=self.config["device"]
            )

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
            "window":1,
            "exog_window": 1,
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

        self.info["model"] = "NARX"
        self.info["model_config"] = self.config

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

        assert self.config['window'] > 0, "window must be greater than 0"
        assert self.config['exog_window'] > 0, "window must be greater than 0"

        assert self.config['lr'] > 0, "lr must be greater than 0"

        assert (self.config['optimizer'] in [
                                            getattr(torch.optim, o) for o in dir(torch.optim) 
                                            if not o.startswith("_")
                                            ]), "optimizer must be part of torch.optim"
        
        assert self.config['criterion'] is not None, "criterion cannot be None"

        assert (self.config['batch_norm_momentum'] >= 0 and 
                 self.config['batch_norm_momentum'] <= 1), "batch_norm_momentum must be between 0 and 1"
        
        assert self.config['dropout'] >= 0 and self.config['dropout'] <= 1, "dropout must be between 0 and 1"
    
    def _prep_training_data(self):
        """Prepares data to use for training"""
        lookback = max(self.config["window"], self.config["exog_window"])

        endog_inputs = torch.zeros((
                                len(self.data)-lookback, 
                                self.config["window"]
                        )
        )
        
        exog_inputs = torch.zeros((
                                len(self.data)-lookback, 
                                self.config["exog_window"]
                        )
        )

        targets = torch.zeros((len(self.data)-lookback, 1))

        for i in range(lookback, len(self.data)):
            endog_features = self.data[i-self.config["window"]:i].clone()
            exog_features = self.exogenous_data[i-self.config["exog_window"]:i].clone()

            endog_inputs[i-lookback, :] = endog_features
            exog_inputs[i-lookback, :] = exog_features
            targets[i-lookback, :] = self.data[i].clone()

        return endog_inputs, exog_inputs, targets 

    def fit(self, epochs=20, verbose=False):
        """
        Trains the model

        Arguments:
        *epochs (int): Number of epochs to train the model.
        *verbose (Optional[bool]): Whether to report loss for each epoch

        Returns:
        *train_hist (list): List of training losses at each epoch
        """
        # Loading data and model to device
        endog_inps, exog_inps, targets = self._prep_training_data()
        self.model = self.model.to(self.config['device'])

        # Training
        train_hist = []
        criterion = self.config['criterion']
        optimizer = self.config['optimizer'](self.model.parameters(), self.config['lr'])
        batch_size = self.config["batch_size"]

        self.fitted = torch.zeros(self.data.shape)*torch.nan

        self.model.train()
        for epoch in range(epochs):
            
            losses = []

            for batch in range(0, len(targets), batch_size):

                if batch + batch_size > len(targets):
                    batch_y = endog_inps[batch:, :].clone().to(self.config["device"])
                    batch_x = exog_inps[batch:, :].clone().to(self.config["device"])
                    batch_target = targets[batch:, :].clone().to(self.config["device"])
                else:
                    batch_y = endog_inps[batch:batch+batch_size, :].clone().to(self.config["device"])
                    batch_x = exog_inps[batch:batch+batch_size, :].clone().to(self.config["device"])
                    batch_target = targets[batch:batch+batch_size, :].clone().to(self.config["device"]) 
                

                optimizer.zero_grad()
                predictions = self.model(batch_y, batch_x)
                loss = criterion(predictions, batch_target)

                loss.backward()
                optimizer.step()

                losses.append(loss.item())
            
            average_loss = sum(losses)/len(losses)
            train_hist.append(average_loss)
            if verbose:
                print(f'Epoch [{epoch+1}/{epochs}] - Loss: {average_loss:.4f}')

        self.model.eval()

        with torch.no_grad():
            fitted_data = self.model(endog_inps, exog_inps)

        self.fitted[-len(fitted_data):] = fitted_data[:,0]

        return train_hist

    def predict(
                self, 
                h,
                exogenous_inputs=None,
):
        """
        Predict the next h values with the trained model

        Arguments:

        *h (int): Forecast horizon, minimum = 1
        *exogenous_inputs (Optional[array_like]): Exogenous values to use
            when forecasting. If future exogenous values are not available;
            will use the last encountered values. Default = None

        Returns:

        *forecasts (array_like): Predicted h values
        """
        endog_features = self.data[-self.config["window"]:].clone()
        exog_features = self.exogenous_data[-self.config["exog_window"]:].clone()

        forecasts = torch.zeros(h)
        self.model.eval()

        for step in range(h):
            with torch.no_grad():
                step_fc = self.model(
                                    endog_features.unsqueeze(0), 
                                    exog_features.unsqueeze(0)
                )

            forecasts[step] = step_fc[0, 0]

            endog_features = torch.cat((endog_features[1:], step_fc[0]), dim=0)

            if exogenous_inputs is not None:
                try:
                    cur_exog = DataLoader(exogenous_inputs).match_dims(1, return_type="tensor")[step]
                    exog_features = torch.cat((exog_features[1:], cur_exog.unsqueeze(0)), dim=0)
                except: # noqa: E722
                    pass
        
        return forecasts