import matplotlib
import matplotlib.style
import matplotlib.pyplot as plt
from chronokit.preprocessing.dataloader import DataLoader
from chronokit.utils.evaluation_utils.metrics import * 

def plot_predictions(y_true, y_pred, bounds=None, pre_vals=None, figsize=(12,8), colors=None, bounds_fill_alpha=0.7, title=None, style=None, metrics=None):
    """
    Utility function for plotting prediction results of the time series model
    
    Arguments:

    *y_true (array_like): Ground truth values
    *y_pred (array_like): Predicted values
    *bounds (Optional[iterable]): Confidence bounds for the prediction interval
    *pre_vals (Optional[array_like]): Values that come before the predicted values i.e; last n-points of the training data
    *figsize (Optional[tuple]): Size of the plot
    *colors (Optional[iterable]): Colors of the lines/points on the plot
    *bounds_fill_alpha (Optional[float]): Alpha for the transparency of the filled values between confidence bounds
    *title (Optional[str]): Title of the plot
    *style (Optional[str]): Style of the plot 'https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html'
    *metrics (Optional[iterable]): Evaluation metrics to use for the predictions to report on the plot
    """
    y_true = DataLoader(y_true).to_numpy()
    y_pred = DataLoader(y_pred).to_numpy()

    use_colors = {"y_true": "blue", "y_pred": "orange", "bounds": "gray", "pre_vals": "black"}

    if colors:
        try:
            iter(colors)
        except TypeError:
            raise TypeError("Provide colors as an iterable")
        
        if type(colors) == dict:
            for key in colors:
                assert(key in list(use_colors.keys())), f"Ensure that keys in colours dictionary are {list(use_colors.keys())}"
                use_colors[key] = colors[key]

        else:
            for ind, c in enumerate(colors):
                use_colors[list(use_colors.keys())[ind]] = c

    error_bounds = None

    if bounds:
        try:
            iter(bounds)
        except TypeError:
            raise TypeError("Provide bounds as an iterable of length 2")
        
        assert (len(bounds) == 2), "Provide bounds as an iterable of length 2"

        if type(bounds) == dict:

            assert (list(bounds.keys()) == ["upper", "lower"]), "Provide bounds dictionary keys as ['upper', 'lower']"

            for val in list(bounds.values()):
                assert (len(val) == len(y_pred)), "Length of bounds must match length of predictions"

            error_bounds = {"upper": DataLoader(bounds["upper"]).to_numpy(), "lower": DataLoader(bounds["lower"]).to_numpy()}

        else:
            for val in bounds:
                assert (len(val) == len(y_pred)), "Length of bounds must match length of predictions"

            error_bounds = {"upper": DataLoader(bounds[0]).to_numpy(), "lower": DataLoader(bounds[1]).to_numpy()}
    
    plt_metrics = None
    if metrics:
        plt_metrics = {}
        try:
            iter(metrics)
            if type(metrics) == dict:
                raise TypeError("Metrics argument cannot be a dictionary")
        except TypeError:
            raise TypeError("Provide metrics as an iterable of length 2")
        
        for i in metrics:
            assert (type(i) == str), "Provide metrics as an iterable with string entries"
            assert (i in ["rmse", "mse", "mae"]), "Supported metrics are: ['rmse', 'mse' and 'mae']"

            if i == "rmse":
                plt_metrics["RMSE"] = rmse(y_pred, y_true).item()
            if i == "mse":
                plt_metrics["MSE"] = mse(y_pred, y_true).item()
            if i == "mae":
                plt_metrics["MAE"] = mae(y_pred, y_true).item()
    
    assert (type(title) == str or title is None), "Plot title must be a string"

    if style:
        assert(type(style) == str), "Provide style as a string"
        matplotlib.style.use(style)

    plt.figure(figsize=figsize)
    if pre_vals is not None:
        pre_vals = DataLoader(pre_vals).to_numpy()
        main_plt_range = range(len(pre_vals), len(pre_vals)+len(y_pred))
        plt.plot(range(len(pre_vals)), pre_vals, color=use_colors["pre_vals"])
    else:
        main_plt_range = range(len(y_pred))
    plt.plot(main_plt_range, y_true, color=use_colors["y_true"], label="Y True")
    plt.plot(main_plt_range, y_pred, color=use_colors["y_pred"], label="Y Predicted")
    if error_bounds:
        plt.fill_between(main_plt_range, error_bounds["upper"], error_bounds["lower"], 
                         color=use_colors["bounds"], alpha=bounds_fill_alpha, label="Prediction Error Bounds")
    if plt_metrics:
        append_title = ""
        for metric in plt_metrics:
            score = plt_metrics[metric]
            append_title += metric + f":{score:.3f} "
        if title:
            title = title + "\n" + append_title
        else:
            title = append_title
    plt.title(title)
    plt.legend(loc="best")
    plt.show()
    