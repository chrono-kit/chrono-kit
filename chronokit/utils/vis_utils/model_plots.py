import numpy as np
import matplotlib
import matplotlib.style
import matplotlib.pyplot as plt
from chronokit.preprocessing._dataloader import DataLoader
from chronokit.utils.evaluation.metrics import (rmse, 
                                                mse, 
                                                mae,
                                                mase,
                                                mape,
                                                symmetric_mape)

def plot_predictions(
    y_true,
    y_pred,
    bounds=None,
    pre_vals=None,
    figsize=(16, 6),
    colors=None,
    bounds_fill_alpha=0.7,
    title=None,
    style=None,
    metrics=None,
    **kwargs
):
    """
    Utility function for plotting prediction results of the time series model

    Arguments:

    *y_true (array_like): Ground truth values
    *y_pred (array_like): Predicted values
    *bounds (Optional[iterable]): Confidence bounds for the prediction interval
    *pre_vals (Optional[array_like]): Values that come before the predicted values i.e;
        last n-points of the training data
    *figsize (Optional[tuple]): Size of the plot
    *colors (Optional[iterable]): Colors of the lines/points on the plot
    *bounds_fill_alpha (Optional[float]): Alpha for the transparency of the filled
        values between confidence bounds
    *title (Optional[str]): Title of the plot
    *style (Optional[str]): Style of the plot
        'https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html'
    *metrics (Optional[iterable]): Evaluation metrics to use for the
        predictions to report on the plot

    **Keyword Arguments:

    *ax (matplotlib.pyplot.ax): Subplot ax to plot the figure. Will not show the plot if given,
        Instead will plot the figure on the ax. Used on .evaluate() methods of the models
    """

    ax = kwargs.get("ax", None)

    try:
        y_true = DataLoader(y_true).match_dims(1, return_type="numpy")
    except:  # noqa: E722
        raise ValueError("Expecting y_true as an array_like")
    
    try:
        y_pred = DataLoader(y_pred).match_dims(1, return_type="numpy")
    except:  # noqa: E722
        raise ValueError("Expecting y_pred as an array_like")

    use_colors = {
        "y_true": "blue",
        "y_pred": "firebrick",
        "bounds": "gray",
        "pre_vals": "black",
    }

    if colors:
        try:
            iter(colors)
        except TypeError:
            raise TypeError("Provide colors as an iterable")

        if isinstance(colors, dict):
            for key in colors:
                assert key in list(
                    use_colors.keys()
                ), f"Ensure that keys in colours dictionary are {list(use_colors.keys())}"
                use_colors[key] = colors[key]

        else:
            for ind, c in enumerate(colors):
                use_colors[list(use_colors.keys())[ind]] = c

    error_bounds = None

    if pre_vals is not None:
        try:
            pre_vals = DataLoader(pre_vals).match_dims(1, return_type="numpy")
        except:  # noqa: E722
            raise ValueError("Expecting pre_vals argument as an array_like")
        
    if bounds:
        try:
            iter(bounds)
        except TypeError:
            raise TypeError("Provide bounds as an iterable of length 2")

        assert len(bounds) == 2, "Provide bounds as an iterable of length 2"

        if isinstance(bounds, dict):
            assert list(bounds.keys()) == [
                "upper",
                "lower",
            ], "Provide bounds dictionary keys as ['upper', 'lower']"

            for val in list(bounds.values()):
                assert len(val) == len(y_pred), "Length of bounds must match length of predictions"

            
            error_bounds = {
                    "upper": DataLoader(bounds["upper"]).match_dims(1, return_type="numpy"),
                    "lower": DataLoader(bounds["lower"]).match_dims(1, return_type="numpy")
                }

        else:
            for val in bounds:
                assert len(val) == len(y_pred), "Length of bounds must match length of predictions"

            
            error_bounds = {
                    "upper": DataLoader(bounds[0]).match_dims(1, return_type="numpy"),
                    "lower": DataLoader(bounds[1]).match_dims(1, return_type="numpy")
                }
        
        if pre_vals is not None:

            error_bounds["upper"] = np.concatenate((pre_vals[-1].reshape(tuple([1 for x in error_bounds["upper"].shape])), error_bounds["upper"]), axis=0)
            error_bounds["lower"] = np.concatenate((pre_vals[-1].reshape(tuple([1 for x in error_bounds["lower"].shape])), error_bounds["lower"]), axis=0)

    plt_metrics = None
    if metrics:
        plt_metrics = {}
        try:
            iter(metrics)
            if isinstance(metrics, dict):
                raise TypeError("Metrics argument cannot be a dictionary")
        except TypeError:
            raise TypeError("Provide metrics as an iterable of length 2")

        for i in metrics:
            assert isinstance(i, str), "Provide metrics as an iterable with string entries"
            
            valid_metrics = {
                        "mae": mae,
                        "mse": mse,
                        "rmse": rmse,
                        "mape": mape,
                        "s_mape": symmetric_mape,
                        "mase": mase,
            }

            assert i in valid_metrics, f"Supported metrics are: {list(valid_metrics.keys())}"


            plt_metrics[i.upper()] = valid_metrics[i](y_pred, y_true).item()

    assert isinstance(title, str) or title is None, "Plot title must be a string"

    if style:
        assert isinstance(style, str), "Provide style as a string"
        matplotlib.style.use(style)
    else:
        matplotlib.style.use("ggplot")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show_plot = True
    else:
        show_plot = False

    if pre_vals is not None:
        pre_vals = DataLoader(pre_vals).match_dims(1, return_type="numpy")
        main_plt_range = range(len(pre_vals)-1, len(pre_vals) + len(y_pred))
        ax.plot(range(len(pre_vals)), pre_vals, color=use_colors["pre_vals"])
        ax.scatter(range(len(pre_vals)), pre_vals, color=use_colors["pre_vals"], s=20)
        y_true = np.concatenate((pre_vals[-1].reshape(tuple([1 for x in y_true.shape])), y_true), axis=0)
        y_pred = np.concatenate((pre_vals[-1].reshape(tuple([1 for x in y_pred.shape])), y_pred), axis=0)
    else:
        main_plt_range = range(len(y_pred))
    ax.plot(main_plt_range, y_true, color=use_colors["y_true"], label="Y True")
    ax.plot(main_plt_range, y_pred, color=use_colors["y_pred"], label="Y Predicted")
    ax.scatter(main_plt_range, y_true, color=use_colors["y_true"], s=20)
    ax.scatter(main_plt_range, y_pred, color=use_colors["y_pred"], s=20)
    if error_bounds:
        ax.fill_between(
            main_plt_range,
            error_bounds["upper"],
            error_bounds["lower"],
            color=use_colors["bounds"],
            alpha=bounds_fill_alpha,
            label="Prediction Error Bounds",
        )
    if plt_metrics:
        append_title = ""
        for metric in plt_metrics:
            score = plt_metrics[metric]
            append_title += metric + f":{score:.3f} "
        if title:
            title = title + "\n" + append_title
        else:
            title = append_title
    ax.set_title(title)
    ax.legend(loc="upper left")
    if show_plot:
        plt.show()

def plot_model_fit(model, 
                   figsize=(16, 6),
                   colors=None,
                   title="True vs. Fitted Data",
                   style=None,
                   metrics=None,
                   **kwargs
):

    assert (hasattr(model, "fitted")), "Please fit the model before calling plot_model_fit()"

    plot_predictions(y_true=model.data, y_pred=model.fitted,
                     bounds=None, pre_vals=None, figsize=figsize,
                     colors=colors, title=title, style=style, metrics=metrics, **kwargs)
