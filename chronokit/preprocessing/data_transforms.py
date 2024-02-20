import numpy as np
import pandas as pd
import torch
import warnings
from scipy.stats import boxcox_normmax
from chronokit.preprocessing._dataloader import DataLoader

def differencing(data, order=1, return_numpy=True):
    """
    Apply differencing to the given data.

    Arguments:

    *data (array_like): Data to perform differencing on.
    *order Optional[int]: Order of differencing
    *return_numpy Optional[bool]: Whether to return a numpy array. Will return numpy.ndarray 
    if given, torch.Tensor otherwise.

    Returns:

    *data (array_like): Differenced data

    Example:

    Differencing of order 1: y'_t = y_t - y_t-1
    """

    if return_numpy:
        data = DataLoader(data).to_numpy()
    else:
        data = DataLoader(data).to_tensor()

    for k in range(order):
        data = data[1:] - data[:-1]

    return data


class DataTransform:
    def __init__(self):
        """Base class for all data transformation class to inherit from"""
        pass

    def _transform_assert(self, loader: DataLoader, axis: int, scales: dict):
        """Make necessary assertions for transforming the data"""
        for scale_key in scales:
            scale_dict = scales[scale_key]

            assert isinstance(scale_dict, dict), f"Provide {scale_key} as a dict"

        assert loader.to_tensor().numpy().size != 0, "Size of data must be > 0"

        if loader.data_type == pd.DataFrame:
            data_names = list(loader.organized_df.columns)
            for scale_dict in list(scales.values()):
                if scale_dict != {}:
                    assert len(list(scale_dict.keys())) == len(
                        data_names
                    ), "Provide same amount of entries as columns"
                    assert set(list(scale_dict.keys())) == set(
                        data_names
                    ), "Provide with the same keys as column names"

        else:
            assert loader.data.ndim <= 2, "Dimension of the data must be <= 2"
            assert axis in [0, 1, -1, -2], f"{axis} is not a valid axis"

            if loader.data.ndim == 1:
                assert axis in [
                    0,
                    -1,
                ], f"{axis} is not a valid axis of data of ndim == 1"

                for scale_key in scales:
                    scale_dict = scales[scale_key]
                    if scale_dict != {}:
                        assert list(scale_dict.keys()) == [
                            scale_key
                        ], "For data with ndim == 1, scales should be\
 provided as {}".format(
                            {f"{scale_key}": "value"}
                        )

            else:
                hmap = {0: "col", 1: "row", -1: "row", -2: "col"}
                data_names = [f"{hmap[axis]}{i}" for i in range(loader.data.shape[1 - axis % 2])]
                
                for scale_dict in list(scales.values()):
                    if scale_dict != {}:
                        assert len(list(scale_dict.keys())) == len(
                            data_names
                        ), f"Provide same amount of entries as {hmap[axis]}s"
                        assert set(list(scale_dict.keys())) == set(
                            data_names
                        ), f"Provide keys as {hmap[axis]}0 for {hmap[axis]} of index 0 etc..."

    def _inv_transform_assert(self, loader: DataLoader, names, scales: dict):
        """Make necessary assertions to inverse transforming the data"""
        assert isinstance(names, list), "Provide names argument as a list"

        try:
            if self.transformed_axis:
                pass
        except NameError:
            raise NameError("Must call '.transform()' before calling '.inverse_transform()'")

        assert loader.to_numpy().size != 0, "Size of data must be > 0"

        transform_data = loader.to_tensor()  # noqa: F841

        if names != []:
            for scale_dict in list(scales.values()):
                for name in names:
                    assert name in list(scale_dict.keys()), f"{name} was not in transformed data"
        else:
            if loader.to_numpy().ndim == 2:
                for scale_dict in list(scales.values()):
                    assert loader.data.shape[1 - self.transformed_axis % 2] == len(
                        list(scale_dict.keys())
                    ), f"Expecting size of axis {self.transformed_axis} = {len(list(scale_dict))}\
                        if the names argument is not provided"


class BoxCox(DataTransform):
    def __init__(self):
        """Box-Cox transformation for time series data"""
        super().__init__()
        self.lambdas = {}

    def transform(self, data, axis=0, lambda_values: dict = {}, return_numpy=True):
        """
        Transform given data with box-cox method

        Arguments:

        *data (array_like): Time series data to transform
        *axis (int): Axis to perform the transformation on
        *lambda_values Optional[dict]: Lambda values for the transformation;
            will be estimated if given as an empty dict
        *return_numpy Optional[bool]: Whether to return a numpy array.
            Will return numpy.ndarray if given, torch.Tensor otherwise.

        Returns:

        *transformed (array_like): Box-Cox transformed data.
        """

        loader = DataLoader(data)
        self._transform_assert(loader, axis, scales={"lambda": lambda_values})

        if loader.data_type == pd.DataFrame:
            data_names = list(loader.organized_df.columns)
            if lambda_values == {}:
                lambda_values = {name: None for name in data_names}
        else:
            if loader.data.ndim == 1:
                if lambda_values == {}:
                    lambda_values = {"data": None}
                data_names = ["data"]
            else:
                hmap = {0: "col", 1: "row", -1: "row", -2: "col"}
                data_names = [f"{hmap[axis]}{i}" for i in range(loader.data.shape[1 - axis % 2])]
                if lambda_values == {}:
                    lambda_values = {name: None for name in data_names}

        self.transformed_axis = axis
        transform_data = loader.to_numpy()
        transformed = np.zeros(transform_data.shape)
        for ind in range(len(data_names)):
            if transform_data.ndim == 1:
                current_data = transform_data.copy()

            elif axis in [0, -2]:
                current_data = transform_data[:, ind]
            else:
                current_data = transform_data[ind, :]

            name = data_names[ind]
            lambd = lambda_values[name]
            boxcoxed = self.__box_cox(current_data, lambd, name)

            if transform_data.ndim == 1:
                transformed = boxcoxed.copy()

            elif axis in [0, -2]:
                transformed[:, ind] = boxcoxed

            else:
                transformed[ind, :] = boxcoxed

        if return_numpy:
            return DataLoader(transformed).to_numpy()

        return DataLoader(transformed).to_tensor()

    def inverse_transform(self, data, names: list = [], return_numpy=True):
        """
        Inverse transform given data

        Arguments:

        *data (array_like): Time series data to inverse transform
        *names Optional[list]: Keys for data to inverse transform
            if partial transformation is desired
        *return_numpy Optional[bool]: Whether to return a numpy array.
            Will return numpy.ndarray if given, torch.Tensor otherwise.

        Returns:

        *transformed (array_like): Inversely transformed data.
        """
        loader = DataLoader(data)

        self._inv_transform_assert(loader, names, scales={"lambda": self.lambdas})

        transform_data = loader.to_numpy()
        transformed = np.zeros(transform_data.shape)

        if names == []:
            names = list(self.lambdas.keys())

        for ind in range(len(names)):
            if transform_data.ndim == 1:
                current_data = transform_data.copy()

            elif self.transformed_axis in [0, -2]:
                current_data = transform_data[:, ind]

            else:
                current_data = transform_data[ind, :]

            name = names[ind]
            lambd = self.lambdas[name]

            inversed = (
                np.exp(current_data)
                if lambd == 0
                else np.power((lambd * current_data + 1), 1 / lambd)
            )

            if transform_data.ndim == 1:
                transformed = inversed.copy()

            elif self.transformed_axis in [0, -2]:
                transformed[:, ind] = inversed

            else:
                transformed[ind, :] = inversed

        if return_numpy:
            return DataLoader(transformed).to_numpy()

        return DataLoader(transformed).to_tensor()

    def __box_cox(self, data, lambd=None, data_name="col0"):
        """Perform the box-cox transformation"""
        box_coxed = DataLoader(data).to_numpy()

        if lambd is not None:
            try:
                float(lambd)
            except:  # noqa: E722
                raise ValueError("Provide lambda value as a float")

        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                lambd = boxcox_normmax(box_coxed, method="mle")

        box_coxed = np.log(box_coxed) if lambd == 0 else (np.power(box_coxed, lambd) - 1) / lambd
        self.lambdas[data_name] = lambd

        return box_coxed


class StandardScaling(DataTransform):
    def __init__(self):
        """Standard Scaling for time series data"""
        self.locations = {}
        self.scales = {}

    def transform(
        self,
        data,
        axis=0,
        locations: dict = {},
        scales: dict = {},
        return_numpy=True,
    ):
        """
        Standard scale the given data

        Arguments:

        *data (array_like): Time series data to transform
        *axis (int): Axis to perform the transformation on
        *locations Optional[dict]: Location values to be used for scaling the data;
            will be taken as the mean if not given
        *scales Optional[dict]: Scale values to be used for scaling the data;
            will be taken as the std if not given
        *return_numpy Optional[bool]: Whether to return a numpy array.
            Will return numpy.ndarray if given, torch.Tensor otherwise.

        Returns:

        *transformed (array_like): Transformed data.
        """

        loader = DataLoader(data)
        self._transform_assert(loader, axis, scales={"location": locations, "scale": scales})

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
                data_names = ["data"]
                if locations == {}:
                    locations = {"data": loader.to_numpy().mean()}
                if scales == {}:
                    scales = {"data": loader.to_numpy().std()}
                    for s in list(scales.values()):
                        if s == 0:
                            raise ValueError("cannot scale data with std = 0")
            else:
                hmap = {0: "col", 1: "row", -1: "row", -2: "col"}
                data_names = [f"{hmap[axis]}{i}" for i in range(loader.data.shape[1 - axis % 2])]
                if locations == {}:
                    locs = loader.to_numpy().mean(axis)
                    locations = {data_names[ind]: locs[ind] for ind in range(len(data_names))}
                if scales == {}:
                    stds = loader.to_numpy().std(axis)
                    scales = {data_names[ind]: stds[ind] for ind in range(len(data_names))}
                    for s in list(scales.values()):
                        if s == 0:
                            raise ValueError("cannot scale data with std = 0")
        self.locations = locations
        self.scales = scales

        self.transformed_axis = axis
        transform_data = loader.to_numpy()
        transformed = np.zeros(transform_data.shape)
        for ind in range(len(data_names)):
            if transform_data.ndim == 1:
                current_data = transform_data.copy()

            elif axis in [0, -2]:
                current_data = transform_data[:, ind]

            else:
                current_data = transform_data[ind, :]

            name = data_names[ind]
            mu = self.locations[name]
            sigma = self.scales[name]

            x = DataLoader(current_data).to_numpy()
            standard_scaled = (x - mu) / sigma

            if transform_data.ndim == 1:
                transformed = standard_scaled.copy()

            elif axis in [0, -2]:
                transformed[:, ind] = standard_scaled

            else:
                transformed[ind, :] = standard_scaled

        if return_numpy:
            return DataLoader(transformed).to_numpy()

        return DataLoader(transformed).to_tensor()

    def inverse_transform(self, data, names: list = [], return_numpy=True):
        """
        Inverse transform given data

        Arguments:

        *data (array_like): Time series data to transform
        *names (Optional[list]): Keys for data to inverse transform
            if partial transformation is desired
        *return_numpy Optional[bool]: Whether to return a numpy array.
            Will return numpy.ndarray if given, torch.Tensor otherwise.

        Returns:

        *transformed (array_like): Inverse transformed data.
        """
        loader = DataLoader(data)

        self._inv_transform_assert(
            loader,
            names,
            scales={"location": self.locations, "scale": self.scales},
        )

        transform_data = loader.to_numpy()
        transformed = np.zeros(transform_data.shape)

        if names == []:
            names = list(self.scales.keys())

        for ind in range(len(names)):
            if transform_data.ndim == 1:
                current_data = transform_data.copy()

            elif self.transformed_axis in [0, -2]:
                current_data = transform_data[:, ind]

            else:
                current_data = transform_data[ind, :]

            name = names[ind]
            mu = self.locations[name]
            sigma = self.scales[name]

            x = DataLoader(current_data).to_numpy()
            inversed = mu + x * sigma

            if transform_data.ndim == 1:
                transformed = inversed.copy()

            elif self.transformed_axis in [0, -2]:
                transformed[:, ind] = inversed

            else:
                transformed[ind, :] = inversed

        if return_numpy:
            return DataLoader(transformed).to_numpy()

        return DataLoader(transformed).to_tensor()


class MinMaxScaling(DataTransform):
    def __init__(self, feature_range=(0, 1)):
        """
        MinMax Scaling for time series data

        Arguments:

        *feature_range (Optional[iterable]): Value bounds for the data to be scaled on
        """
        try:
            iter(feature_range)
        except TypeError:
            raise TypeError("Provide feature_range as an iterable")

        assert len(feature_range) == 2, "Provide feature_range as an iterable of length 2"

        self.lb, self.ub = feature_range

        assert self.ub > self.lb, "Provide feature_range as (a,b) where b > a"

        self.mins = {}
        self.maxes = {}

    def transform(self, data, axis=0, return_numpy=True, mins={}, maxes={}):
        """
        MinMax scale the given data

        Arguments:

        *data (array_like): Time series data to transform
        *axis (int): Axis to perform the transformation on
        *return_numpy Optional[bool]: Whether to return a numpy array.
            Will return numpy.ndarray if given, torch.Tensor otherwise.
        *mins Optional[dict]: Minimum values to use for transforming the data
        *maxes Optional[dict]: Maximum values to use for transforming the data

        Returns:

        *transformed: Transformed data.
        """

        loader = DataLoader(data)
        self._transform_assert(loader, axis, scales={})

        if loader.data_type == pd.DataFrame:
            data_names = list(loader.organized_df.columns)
            if mins == {}:
                mins = {name: loader.organized_df[name].min() for name in data_names}
            if maxes == {}:
                maxes = {name: loader.organized_df[name].max() for name in data_names}

            for key in mins:
                if mins[key] == maxes[key]:
                    raise ValueError("Cannot scale with min(data)=max(data)")

        else:
            if loader.data.ndim == 1:
                data_names = ["data"]
                if mins == {}:
                    mins = {"data": loader.to_numpy().min()}
                if maxes == {}:
                    maxes = {"data": loader.to_numpy().max()}

                if mins["data"] == maxes["data"]:
                    raise ValueError("Cannot scale with min(data)=max(data)")

            else:
                hmap = {0: "col", 1: "row", -1: "row", -2: "col"}
                data_names = [f"{hmap[axis]}{i}" for i in range(loader.data.shape[1 - axis % 2])]

                if mins == {}:
                    mins_ = loader.to_numpy().min(axis)
                    mins = {data_names[ind]: mins_[ind] for ind in range(len(data_names))}
                if maxes == {}:
                    maxes_ = loader.to_numpy().max(axis)
                    maxes = {data_names[ind]: maxes_[ind] for ind in range(len(data_names))}

                for key in mins:
                    if mins[key] == maxes[key]:
                        raise ValueError("Cannot scale with min(data)=max(data)")

        self.mins = mins
        self.maxes = maxes

        self.transformed_axis = axis
        transform_data = loader.to_numpy()
        transformed = np.zeros(transform_data.shape)
        for ind in range(len(data_names)):
            if transform_data.ndim == 1:
                current_data = transform_data.copy()

            elif axis in [0, -2]:
                current_data = transform_data[:, ind]

            else:
                current_data = transform_data[ind, :]

            name = data_names[ind]
            xmin = self.mins[name]
            xmax = self.maxes[name]

            x = DataLoader(current_data).to_numpy()
            minmax_scaled = (x - xmin) / (xmax - xmin)
            minmax_scaled = minmax_scaled * (self.ub - self.lb) + self.lb

            if transform_data.ndim == 1:
                transformed = minmax_scaled.copy()

            elif axis in [0, -2]:
                transformed[:, ind] = minmax_scaled

            else:
                transformed[ind, :] = minmax_scaled

        if return_numpy:
            return DataLoader(transformed).to_numpy()

        return DataLoader(transformed).to_tensor()

    def inverse_transform(self, data, names: list = [], return_numpy=True):
        """
        Inverse transform given data

        Arguments:

        *data (array_like): Time series data to transform
        *names (Optional[list]): Keys for data to inverse transform
            if partial transformation is desired
        *return_numpy Optional[bool]: Whether to return a numpy array.
            Will return numpy.ndarray if given, torch.Tensor otherwise.

        Returns:

        *transformed Optional[bool]: Inverse transformed data.
        """

        loader = DataLoader(data)

        self._inv_transform_assert(loader, names, scales={})

        transform_data = loader.to_numpy()
        transformed = np.zeros(transform_data.shape)

        if names == []:
            names = list(self.mins.keys())

        for ind in range(len(names)):
            if transform_data.ndim == 1:
                current_data = transform_data.copy()

            elif self.transformed_axis in [0, -2]:
                current_data = transform_data[:, ind]

            else:
                current_data = transform_data[ind, :]

            name = names[ind]
            xmin = self.mins[name]
            xmax = self.maxes[name]

            x = DataLoader(current_data).to_numpy()
            inversed = (x - self.lb) / (self.ub - self.lb)
            inversed = inversed * (xmax - xmin) + xmin
            if transform_data.ndim == 1:
                transformed = inversed.copy()

            elif self.transformed_axis in [0, -2]:
                transformed[:, ind] = inversed

            else:
                transformed[ind, :] = inversed

        if return_numpy:
            return DataLoader(transformed).to_numpy()

        return DataLoader(transformed).to_tensor()
