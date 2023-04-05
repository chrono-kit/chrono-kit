import matplotlib
import matplotlib.pyplot as plt

def plot_train_test_split(train_data, test_data, val_data=[], var="", title="Train Test Split"):
    
    """"Plots train and test data in one plot for univariate data.
        Parameters:
            train_data : Training data possibly of type pd.DataFrame
            test_data  : Test data possibly of type pd.Dataframe
            val_data   : Validation data possibly of type pd.Dataframe
            var        : Univariable's name which will be plotted 
            title      : Title of the plot"""
    
    if(var == ""):
        print("No variable name is provided")
    elif(val_data == []):
        plt.plot(range(len(train_data)), train_data[var], label="Train")
        plt.plot(range(len(train_data), len(train_data+test_data)), test_data[var], label="Test")
        plt.legend(loc="best")
        plt.title(title)
        plt.show
    else:
        plt.plot(range(len(train_data)), train_data[var], label="Train")
        plt.plot(range(len(train_data), len(train_data+test_data+val_data)), test_data[var], label="Test")
        plt.plot(range(len(train_data), len(train_data+test_data+val_data)), val_data[var], label="Test")
        plt.legend(loc="best")
        plt.title(title)
        plt.show

def plot_pred_vs_label(ground_truth, prediction, title = "Predictions vs Ground Truth"):

    """"Plots predictions vs true labels graph for univariate predictions.
        Parameters:
            ground_truth : True values of the predicted variable possibly of type pd.DataFrame or pd.Series
            prediction   : Predicted data of the variable, possibly of type pd.Dataframe or pd.Series
            title        : Title of the plot"""

    plt.plot(range(len(ground_truth)), ground_truth, label="Ground Truth")
    plt.plot(range(len(prediction)), prediction, label = "Predictions")
    plt.legend(loc="best")
    plt.title(title)
    plt.show