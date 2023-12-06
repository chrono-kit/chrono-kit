import torch
import torch.nn as nn
from chronokit.base._models import Model
from chronokit.preprocessing._dataloader import DataLoader


class NARXModel(torch.nn.Module):

    def __init__(self, n_features, hidden_size):
        super(NARXModel, self).__init__()
        self.Wx = nn.Linear(n_features, hidden_size, bias=False)
        self.Wy = nn.Linear(1, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x, y):
        x_out = self.Wx(x)
        y_out = self.Wy(y)
        output = self.tanh(x_out + y_out)
        output = self.output_layer(output)
        return output

class NARX(Model):

    def __init__(self,x,y,batch_size,hidden_size,device,n_features,window=1):
        # Set allowed kwargs
        self.set_allowed_kwargs(['window', 'n_features' , 'batch_size', 'hidden_size', 'device'])

        # Set attributes
        self.x = DataLoader(x).to_tensor()
        self.y = DataLoader(y).to_tensor()
        self.window = window
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.device = device
        self.model = NARXModel(n_features, hidden_size)

    def _check_dimensions(self, x, y):
        # If 1D, convert to 2D
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)

        return x, y
    
    def _load_data(self):
        # Checking dimensions
        self.x, self.y = self._check_dimensions(self.x, self.y)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(self.x, self.y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,shuffle=False,drop_last=True)
        return dataloader
    
    def fit(self, epochs=20, lr=0.001):
        # Loading data and model to device
        self.dataloader = self._load_data()
        self.model = self.model.to(self.device)

        # Training
        train_hist = []
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(),lr=0.01)
        total_loss = 0.0

        for epoch in range(epochs):
            self.model.train()
            for batch_X, batch_y in self.dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                predictions = self.model(batch_X,batch_y)
                loss = criterion(predictions, batch_y)

                loss.backward()
                optimizer.step()
        
                total_loss += loss.item()

            # Calculate average training loss and accuracy
            average_loss = total_loss / len(self.dataloader)
            train_hist.append(average_loss)
            print(f'Epoch [{epoch+1}/{epochs}] - Training Loss: {average_loss:.4f}')

        return train_hist


    def predict(self, x_input, y_input):
        # Expected input: (prediction_batch_size, n_features)
        # Expected output: (prediction_batch_size, 1)
        # Convert to tensor if not already
        if type(x_input) != torch.Tensor and type(y_input) != torch.Tensor:
            x_input = torch.tensor(x_input).float().to(self.device)
            y_input = torch.tensor(y_input).float().to(self.device)

        self.model.eval()
        pred = self.model(x_input, y_input)
        return pred