import numpy as np
import pandas as pd
import torch
import warnings

class DataLoader:

    def __init__(self, data):

        """ A class to transform given data into desirable types
            Currently accepted data types are: 'pd.DataFrame', 'pd.Series', 'np.ndarray', 'torch.Tensor"""
        
        self.accepted_types = [pd.DataFrame,
                               pd.Series,
                               np.ndarray,
                               torch.Tensor,
                               list]
        
        self.data_type = type(data)
        
        assert (self.data_type in self.accepted_types), f"{type(data).__name__} is not an accepted data type"

        if self.data_type == pd.DataFrame:

            self.original_df = data
            self.data = self.__organize_df(data)
            if self.operations != []:
                warnings.warn(f"Some changes were performed in the dataset.\n{self.operations}", stacklevel=2)
        
        else:
            self.data = data
    
    def __organize_df(self, df: pd.DataFrame):

        """ Method for organizing a possibly unorganized dataframe while also making sure all entries are convertible to tensors
            and keeping track of the operations done on the original dataframe"""

        self.organized_df = df.copy()

        self.operations = []

        self.dates = None
        self.data_columns = list(df.columns)

        for ind, dtype in enumerate(df.dtypes):
            col = df.dtypes.index[ind]

            if dtype == object:
                try:
                    self.dates = pd.to_datetime(df[col])
                    self.operations.append(f"Turned entries of column '{col}' into datetime")

                    self.data_columns.remove(col)
                    self.organized_df.pop(col)
                    self.organized_df = self.organized_df.set_index(self.dates)
                    self.organized_df.index.name = "Dates"
                    self.operations.append("Set dates as an index")
                    
                except:
                    
                    try:
                        float_vals = df[col].values.astype(np.float32)
                        self.organized_df[col] = float_vals
                        self.operations.append(f"Turned {col} entries into floats")
                    except:
                        raise Exception(f"Could not handle entries of column: '{col}' ")
        
        return self.organized_df.values

    def to_tensor(self):

        """Turn self.data into tensors"""

        if type(self.data) == torch.Tensor:
            return torch._cast_Float(self.data.detach().clone())
        
        else:
            return torch.tensor(self.data, dtype=torch.float32).detach().clone()
        
    def to_numpy(self):

        """Turn self.data into numpy arrays"""

        if type(self.data) == np.ndarray:
            return np.float32(self.data.copy())
        
        else:
            return np.array(self.data, dtype=np.float32).copy()
        