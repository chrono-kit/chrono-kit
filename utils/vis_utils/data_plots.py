import matplotlib
import matplotlib.style
import matplotlib.pyplot as plt
from dataloader import DataLoader

def plot_decomp(trend, seasonal, remainder, figsize=(12,8), colors=None, style=None):

    if style:
        assert(type(style) == str), "Provide style as a string"
        matplotlib.style.use(style)
    
    use_colors = {"trend": "blue", "seasonal": "blue", "remainder": "blue"}

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

    trend = DataLoader(trend).to_numpy()
    seasonal = DataLoader(seasonal).to_numpy()
    remainder = DataLoader(remainder).to_numpy()
    
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    ax1, ax2, ax3 = axes     
    ax1.plot(range(len(trend)), trend, color=use_colors["trend"])
    ax1.set_ylabel("Trend")
    ax2.plot(range(len(seasonal)), seasonal, color=use_colors["seasonal"])
    ax2.set_ylabel("Seasonal")
    ax3.scatter(range(len(remainder)), remainder, color=use_colors["remainder"])
    ax3.set_ylabel("Remainder")
    plt.show()
        

def plot_train_test_split(train_data, test_data, val_data=None, figsize=(12,8), title: str =None, colors=None, style=None):
    
    """"Plots train and test data in one plot for univariate data.
        Parameters:
            train_data : Training data possibly of type pd.DataFrame
            test_data  : Test data possibly of type pd.Dataframe
            val_data   : Validation data possibly of type pd.Dataframe
            title      : Title of the plot"""
    
    assert (type(title) == "str" or title is None), "Plot title must be a string"

    if style:
        assert(type(style) == str), "Provide style as a string"
        matplotlib.style.use(style)
    
    use_colors = {"train": "blue", "val": "orange", "test": "red"}

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

    train = DataLoader(train_data).to_numpy()
    test = DataLoader(test_data).to_numpy()

    if val_data:
        val = DataLoader(val_data).to_numpy()
        plt.figure(figsize=figsize)
        plt.plot(range(len(train)), train, label="Train", color=use_colors["train"])
        plt.plot(range(len(train), len(train+val_data)), val, label="Validation", color=use_colors["val"])
        plt.plot(range(len(train+val_data), len(train+val_data+test_data)), test, label="Test", color=use_colors["test"])
        plt.legend(loc="best")
        plt.title(title)
        plt.show()
       
    else:
        plt.figure(figsize=figsize)
        plt.plot(range(len(train)), train, label="Train", color=use_colors["train"])
        plt.plot(range(len(train), len(train+test)), test, label="Test", color=use_colors["test"])
        plt.legend(loc="best")
        plt.title(title)
        plt.show()
    