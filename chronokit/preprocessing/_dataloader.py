import numpy as np
import pandas as pd
import torch
import warnings


class DataLoader:
    def __init__(self, data):
        """
        A class to transform given data into desirable types
        
        Currently accepted data types are: 
            'pd.DataFrame', 'pd.Series', 
            'np.ndarray', 'torch.Tensor', 'list'
            'np.number' 'float', 'int'
        """

        self.accepted_types = [
            pd.DataFrame,
            pd.Series,
            np.ndarray,
            torch.Tensor,
            list,
        ]

        self.data_type = type(data)

        if self.data_type not in self.accepted_types:
            if not self.is_valid_numeric(data):
                raise TypeError(f"{self.data_type.__name__} is not an accepted data type")
        
        #TODO: Check for nan,inf etc.. values in given array_like data

        if self.data_type == pd.DataFrame:
            self.original_df = data
            self.data = self.__organize_df(data)
            if self.operations != []:
                warnings.warn(
                    f"Some changes were performed in the dataset.\n{self.operations}",
                    stacklevel=2,
                )

        else:
            self.data = data
    
    def is_valid_numeric(self, value):
        if value is None:
            return False
        
        #check valid numeric numpy number
        if isinstance(value, np.number) and not (np.isinf(value) or np.isnan(value)):
            return True
        
        #Check valid numeric tensor
        if torch.is_tensor(value):
            if value.ndim == 0 and not (torch.isinf(value) or torch.isnan(value)):
                return True
        
        #check if valid built-in numeric
        if isinstance(value, (int, float)) and value not in (float('inf'), float('-inf'), float('nan')):
            return True
            
        return False
    
    def __organize_df(self, df: pd.DataFrame):
        """Method for organizing a possibly unorganized dataframe while also making sure
        all entries are convertible to tensors and keeping track of the operations done
        on the original dataframe"""

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

                except:  # noqa: E722
                    try:
                        float_vals = df[col].values.astype(np.float32)
                        self.organized_df[col] = float_vals
                        self.operations.append(f"Turned {col} entries into floats")
                    except:  # noqa: E722
                        raise Exception(f"Could not handle entries of column: '{col}' ")

        return self.organized_df.values

    def to_tensor(self):
        """Turn self.data into tensors"""

        if isinstance(self.data, torch.Tensor):
            return torch._cast_Float(self.data.detach().clone())

        else:
            return torch.tensor(self.data, dtype=torch.float32).detach().clone()

    def to_numpy(self):
        """Turn self.data into numpy arrays"""

        if isinstance(self.data, np.ndarray):
            return np.float32(self.data.copy())

        else:
            return np.array(self.data, dtype=np.float32).copy()
    
    def match_dims(self, dims, return_type="numpy"):
        """Match the number of dimensions of self.data to specified dims"""

        assert(dims > 0), "dims must be greater than 0"

        data = self.to_numpy()

        if data.ndim == dims:
            if return_type == "numpy":
                return data
            else:
                return self.to_tensor()
        
        elif data.ndim < dims:
            data = np.expand_dims(data, axis=tuple([-x-1 for x in range(dims-data.ndim)]))
        
        elif data.ndim > dims:
            one_axes = np.where(np.array(data.shape) == 1)[0]
            assert (dims <= data.ndim - len(one_axes)), f".match_dims cannot be used where \
                dims > (data.ndim - number of axes with shape 1).\n\
                Got {dims} > {data.ndim - len(one_axes)}"

            data = np.squeeze(data, axis=tuple([x for x in one_axes[list(range(data.ndim-dims))]]))
        
        if return_type == "numpy":
            return data
        else:
            return torch.tensor(data, dtype=torch.float32).detach().clone()
            



