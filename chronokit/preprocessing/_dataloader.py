import numpy as np
import pandas as pd
import torch
import warnings


class DataLoader:
    def __init__(self, data):
        """A class to transform given data into desirable types
        Currently accepted data types are: 'pd.DataFrame', 'pd.Series', 'np.ndarray', 'torch.Tensor
        """

        self.accepted_types = [
            pd.DataFrame,
            pd.Series,
            np.ndarray,
            torch.Tensor,
            list,
        ]

        self.data_type = type(data)

        if "float" not in self.data_type.__name__:
            assert (
                self.data_type in self.accepted_types
            ), f"{type(data).__name__} is not an accepted data type"

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

        if len(data.shape) == dims:
            if return_type == "numpy":
                return self.to_numpy()
            else:
                return self.to_tensor()

        one_axes = np.where(np.array(data.shape) == 1)[0]
        assert (one_axes == data.ndim - 1), ".match_dims cannot be called on a data with multiple dimensions with shape != 1"

        if data.ndim == dims:
            if return_type == "numpy":
                return data
            else:
                return self.to_tensor()
        
        elif data.ndim < dims:
            data = np.expand_dims(data, axis=(-x-1 for x in range(dims-data.ndim)))
        
        elif data.ndim > dims:
            data = np.squeeze(data, axis=[x for x in one_axes[-dims:]])
        
        if return_type == "numpy":
            return data
        else:
            return torch.tensor(data, dtype=torch.float32).detach().clone()
            



