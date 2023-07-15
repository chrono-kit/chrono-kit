import numpy as np
import pandas as pd
import torch
from .dataloader import DataLoader
from scipy.stats import boxcox_normmax

class DataTransform:

    def __init__(self):
        """Base class for all data transformation class to inherit from"""
        pass

    def transform_assert(self, loader: DataLoader, axis: int, scales: dict):
        """Make necessary assertions for transforming the data"""
        for scale_key in scales:

            scale_dict = scales[scale_key]

            assert(type(scale_dict) == dict), f"Provide {scale_key} as a dict"

        assert (loader.to_tensor().numpy().size != 0), "Size of data must be > 0"

        if loader.data_type == pd.DataFrame:
            data_names = list(loader.organized_df.columns)
            for scale_dict in list(scales.values()):
                
                if scale_dict != {}:                
                    assert (len(list(scale_dict.keys())) == len(data_names)), f"Provide same amount of entries as columns"
                    assert (set(list(scale_dict.keys())) == set(data_names)), "Provide with the same keys as column names"

        else:
            assert (loader.data.ndim <= 2), "Dimension of the data must be <= 2"
            assert (axis in [0, 1, -1, -2]), f"{axis} is not a valid axis"

            if loader.data.ndim == 1:
                assert (axis in [0,-1]), f"{axis} is not a valid axis of data of ndim == 1"
                
                for scale_key in scales:    
                    scale_dict = scales[scale_key]
                    if scale_dict != {}:
                        assert (list(scale_dict.keys()) == [scale_key]), "For data with ndim == 1, scales should be\
 provided as {}".format({f'{scale_key}': 'value'})

            else:
                hmap = {0: "row", 1: "col", -1: "col", -2: "row"}
                data_names = [f"{hmap[axis]}{i}" for i in range(loader.data.shape[1-  axis % 2])]

                for scale_dict in list(scales.values()):
                    if scale_dict != {}:
                        assert (len(list(scale_dict.keys())) == len(data_names)), f"Provide same amount of entries as {hmap[axis]}s"
                        assert (set(list(scale_dict.keys())) == set(data_names)), f"Provide keys as {hmap[axis]}0 for {hmap[axis]} of index 0 etc..."

    def inv_transform_assert(self, loader: DataLoader, names, scales: dict):
        """Make necessary assertions to inverse transforming the data"""
        assert(type(names) == list), "Provide names argument as a list"

        try:
            if self.transformed_axis:
                pass
        except NameError:
            raise NameError("Must call '.transform()' before calling '.inverse_transform()'")

        assert(type(names) == list), "Provide names argument as a list"

        assert (loader.to_tensor().numpy().size != 0), "Size of data must be > 0"

        transform_data = loader.to_tensor()

        if names != []:
            for scale_dict in list(scales.values()):
                for name in names:
                    assert(name in list(scale_dict())), f"{name} was not in transformed data"
        else:
            assert (len(loader.to_tensor().shape) == 2), "Data must be 2 dimensional if the names argument is not provided"
            for scale_dict in list(scales.values()):
                assert (loader.data.shape[1-  self.transformed_axis % 2] == len(list(scale_dict.keys()))), f"Expecting\
 size of axis {self.transformed_axis} = {len(list(scale_dict))} if the names argument is not provided"


class BoxCox(DataTransform):
    def __init__(self):
        """Box-Cox transformation for time series data"""
        super().__init__()
        self.lambdas = {}

    def transform(self, data, axis=0, lambda_values: dict = {}):
        """
        Transform given data with box-cox method

        Arguments:

        *data (array_like): Time series data to transform
        *axis (int): Axis to perform the transformation on
        *lambda_values (Optional[dict]): Lambda values for the transformation; will be estimated if given as an empty dict 
        """

        loader = DataLoader(data)
        self.transform_assert(loader, axis, scales={"lambda": lambda_values})

        if loader.data_type == pd.DataFrame:
            data_names = list(loader.organized_df.columns)
            if lambda_values == {}:
                lambda_values = {name: None for name in data_names}
        else:
            if loader.data.ndim == 1:
                if lambda_values == {}:
                    lambda_values = {"lambda": None}
            else:
                hmap = {0: "col", 1: "row", -1: "row", -2: "col"}
                data_names = [f"{hmap[axis]}{i}" for i in range(loader.data.shape[1-  axis % 2])]
                if lambda_values == {}:
                    lambda_values = {name: None for name in data_names}

        self.transformed_axis = axis
        transform_data = loader.to_tensor()
        transformed = np.zeros(transform_data.shape)
        for ind in range(len(data_names)):
            if transform_data.ndim == 1:
                current_data = transform_data

            elif axis in [0,-2]:
                current_data = transform_data[:, ind]
            else:
                current_data = transform_data[ind, :]
            
            name = data_names[ind]
            lambd = lambda_values[name]
            boxcoxed = self.__box_cox(current_data, lambd, name)

            if transform_data.ndim == 1:
                transformed = boxcoxed

            elif axis in [0,-2]:
                transformed[:, ind] = boxcoxed
            
            else:
                transformed[ind, :] = boxcoxed

        return transformed
    
    def inverse_transform(self, data, names: list = []):
        """
        Inverse transform given data

        Arguments:

        *data (array_like): Time series data to inverse transform
        *names (Optional[list]): Keys for data to inverse transform if partial transformation is desired  
        """
        loader = DataLoader(data)

        self.inv_transform_assert(loader, names, scales={"lambda": self.lambdas})

        transform_data = loader.to_tensor()
        transformed = np.zeros(transform_data.shape)

        if names == []:
            names = list(self.lambdas.keys())

        for ind in range(len(names)):

            if transform_data.ndim == 1:
                current_data = transform_data

            elif self.transformed_axis in [0,-2]:
                current_data = transform_data[:, ind]
            
            else:
                current_data = transform_data[ind, :]
            
            name = names[ind]
            lambd = self.lambdas[name]

            inversed = torch.exp(current_data) if lambd == 0 else torch.pow(torch.add(torch.mul(lambd, current_data), 1), 1/lambd)

            if transform_data.ndim == 1:
                transformed = inversed
            
            elif self.transformed_axis in [0,-2]:
                transformed[:, ind] = inversed
            
            else:
                transformed[ind, :] = inversed

        return transformed
          
    def __box_cox(self, data, lambd=None, data_name="col0"):
        """Perform the box-cox transformation"""
        box_coxed = data.detach().clone()

        if lambd:
            assert(type(lambd) == float or type(lambd) == int), "Provide lambda value as a float"

        else:  
            lambd = boxcox_normmax(box_coxed.numpy())
        
        box_coxed = torch.log(box_coxed) if lambd == 0 else torch.div(torch.sub(torch.pow(box_coxed, lambd), 1), lambd)
        self.lambdas[data_name] = lambd

        return box_coxed

class StandardScaling(DataTransform):
    def __init__(self):
        """Standard Scaling for time series data"""
        self.locations = {}
        self.scales = {}

    def transform(self, data, axis=0, locations: dict = {}, scales: dict = {}):
        """
        Standard scale the given data

        Arguments:

        *data (array_like): Time series data to transform
        *axis (int): Axis to perform the transformation on
        *locations (Optional[dict]): Location values to be used for scaling the data; will be taken as the mean if not given
        *scales (Optional[dict]): Scale values to be used for scaling the data; will be taken as the std if not given
        """
        loader = DataLoader(data)
        self.transform_assert(loader, axis, scales={"location": locations, "scale": scales})

        if loader.data_type == pd.DataFrame:
            data_names = list(loader.organized_df.columns)
            if locations == {}:
                locations = {name: loader.organized_df[name].mean() for name in data_names}
            if scales == {}:
                scales = {name: loader.organized_df[name].std() for name in data_names}
                for s in list(scales.values()):
                    if s == 0:
                        raise ValueError("cannot scale data with std = 0")
        else:
            if loader.data.ndim == 1:
                if locations == {}:
                    locations = {"loc": loader.to_tensor().mean().item()}
                if scales == {}:
                    scales = {"scale": loader.to_tensor().std().item()}
                    for s in list(scales.values()):
                        if s == 0:
                            raise ValueError("cannot scale data with std = 0")
            else:
                hmap = {0: "row", 1: "col", -1: "col", -2: "row"}
                data_names = [f"{hmap[axis]}{i}" for i in range(loader.data.shape[1-  axis % 2])]
                if locations == {}:
                    locs = loader.to_tensor().mean(axis)
                    locations = {data_names[ind]: locs[ind].item() for ind in range(len(data_names))}
                if scales == {}:
                    stds = loader.to_tensor().std(axis)
                    scales = {data_names[ind]: stds[ind].item() for ind in range(len(data_names))}
                    for s in list(scales.values()):
                        if s == 0:
                            raise ValueError("cannot scale data with std = 0")
        self.locations = locations
        self.scales = scales

        self.transformed_axis = axis
        transform_data = loader.to_tensor()
        transformed = np.zeros(transform_data.shape)
        for ind in range(len(data_names)):

            if transform_data.ndim == 1:
                current_data = transform_data

            elif axis in [0,-2]:
                current_data = transform_data[:, ind]
            
            else:
                current_data = transform_data[ind, :]
            
            name = data_names[ind]
            mu = self.locations[name]
            sigma = self.scales[name]
            
            x = current_data.detach().clone()
            standard_scaled = torch.div(torch.sub(x,mu), sigma)

            if transform_data.ndim == 1:
                transformed = standard_scaled

            elif axis in [0,-2]:
                transformed[:, ind] = standard_scaled
            
            else:
                transformed[ind, :] = standard_scaled

        return transformed
    
    def inverse_transform(self, data, names: list = []):
        """
        Inverse transform given data

        Arguments:

        *data (array_like): Time series data to transform
        *names (Optional[list]): Keys for data to inverse transform if partial transformation is desired  
        """
        loader = DataLoader(data)

        self.inv_transform_assert(loader, names, scales={"location": self.locations, "scale": self.scales})

        transform_data = loader.to_tensor()
        transformed = np.zeros(transform_data.shape)

        for ind in range(len(names)):

            if transform_data.ndim == 1:
                current_data = transform_data

            elif self.transformed_axis in [0,-2]:
                current_data = transform_data[:, ind]
            
            else:
                current_data = transform_data[ind, :]
            
            name = names[ind]
            mu = self.locations[name]
            sigma = self.scales[name]

            x = current_data.detach().clone()
            inversed = torch.add(torch.mul(x, sigma), mu)

            if transform_data.ndim == 1:
                transformed = inversed
            
            elif self.transformed_axis in [0,-2]:
                transformed[:, ind] = inversed
            
            else:
                transformed[ind, :] = inversed

        return transformed

class MinMaxScaling(DataTransform):
    def __init__(self, feature_range=(0,1)):
        """
        MinMax Scaling for time series data

        Arguments:

        *feature_range (Optional[iterable]): Value bounds for the data to be scaled on
        """
        try:
            iter(feature_range)
        except TypeError:
            raise TypeError("Provide feature_range as an iterable")

        assert(len(feature_range) == 2), "Provide feature_range as an iterable of length 2"

        self.lb, self.ub = feature_range

        assert(self.ub > self.lb), "Provide feature_range as (a,b) where b > a"

        self.mins = {}
        self.maxes = {}

    def transform(self, data, axis=0):
        """
        MinMax scale the given data

        Arguments:

        *data (array_like): Time series data to transform
        *axis (int): Axis to perform the transformation on
        """
        loader = DataLoader(data)
        self.transform_assert(loader, axis, scales={})

        if loader.data_type == pd.DataFrame:
            data_names = list(loader.organized_df.columns)
            mins = {name: loader.organized_df[name].min() for name in data_names}
            maxes = {name: loader.organized_df[name].max() for name in data_names}

            for key in mins:

                if mins[key] == maxes[key]:
                    raise ValueError("Cannot scale with min(data)=max(data)")

        else:
            if loader.data.ndim == 1:

                mins = {"min": loader.to_numpy().min()}
                maxes = {"max": loader.to_numpy().max()}

                if mins["min"] == maxes["max"]:
                    raise ValueError("Cannot scale with min(data)=max(data)")
                
            else:
                hmap = {0: "row", 1: "col", -1: "col", -2: "row"}
                data_names = [f"{hmap[axis]}{i}" for i in range(loader.data.shape[1-  axis % 2])]
                
                mins_ = loader.to_tensor().min(axis)
                mins = {data_names[ind]: mins_[ind] for ind in range(len(data_names))}
                maxes_ = loader.to_tensor().max(axis)
                maxes = {data_names[ind]: maxes_[ind] for ind in range(len(data_names))}

                for key in mins:

                    if mins[key] == maxes[key]:
                        raise ValueError("Cannot scale with min(data)=max(data)")
                    
        self.mins = mins
        self.maxes = maxes

        self.transformed_axis = axis
        transform_data = loader.to_tensor()
        transformed = np.zeros(transform_data.shape)
        for ind in range(len(data_names)):

            if transform_data.ndim == 1:
                current_data = transform_data

            elif axis in [0,-2]:
                current_data = transform_data[:, ind]
            
            else:
                current_data = transform_data[ind, :]
            
            name = data_names[ind]
            xmin = self.mins[name]
            xmax = self.maxes[name]
            
            x = current_data.detach().clone()
            minmax_scaled = torch.div(torch.sub(x,xmin), torch.sub(xmax, xmin))
            minmax_scaled = torch.add(torch.mul(minmax_scaled, (self.ub-self.lb)), self.lb)

            if transform_data.ndim == 1:
                transformed = minmax_scaled

            elif axis in [0,-2]:
                transformed[:, ind] = minmax_scaled
            
            else:
                transformed[ind, :] = minmax_scaled

        return transformed

    def inverse_transform(self, data, names: list = []):
        """
        Inverse transform given data

        Arguments:

        *data (array_like): Time series data to transform
        *names (Optional[list]): Keys for data to inverse transform if partial transformation is desired  
        """
        loader = DataLoader(data)

        self.inv_transform_assert(loader, names, scales={})

        transform_data = loader.to_tensor()
        transformed = np.zeros(transform_data.shape)

        for ind in range(len(names)):

            if transform_data.ndim == 1:
                current_data = transform_data

            elif self.transformed_axis in [0,-2]:
                current_data = transform_data[:, ind]
            
            else:
                current_data = transform_data[ind, :]
            
            name = names[ind]
            xmin = self.mins[name]
            xmax = self.maxes[name]

            x = current_data.detach().clone()
            inversed = torch.div(torch.add(x, self.lb), (self.ub-self.lb))
            inversed = torch.add(torch.mul(inversed, torch.sub(xmax-xmin)), xmin)

            if transform_data.ndim == 1:
                transformed = inversed
            
            elif self.transformed_axis in [0,-2]:
                transformed[:, ind] = inversed
            
            else:
                transformed[ind, :] = inversed