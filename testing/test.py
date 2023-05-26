import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from metrics import Metric
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
mpl.rcParams["figure.figsize"] = (12,12)


class Test:

    def __init__(self, root_path):

        self.root = root_path

        self.__get_data()

    
    def __get_data(self):

        frames = [data for data in os.listdir(f"{self.root}/datasets") if data[-3:] == "csv"]


        self.data = {}

        for csv in frames:
            
            df = pd.read_csv(f"{self.root}/datasets/{csv}")
            
            df = df[[df.columns[-1]]] if csv != "GOOG.csv" else df[["Close"]]

            self.data[csv.removesuffix(".csv")] = MinMaxScaler((1,2)).fit_transform(df)
    
    def __plot_session(self, imp_outs, sm_outs):
        
        fig, axes = plt.subplots(len(imp_outs), 1)
        fig.figsize = (12,8)

        for ind, ax in enumerate(axes):
            
            ax.plot(imp_outs[ind], label="Implementation")
            ax.plot(sm_outs[ind], label="Statsmodels")
            ax.legend(loc="best")
            ax.set_title(f"iter_{ind}")
            ax.set_xticks([])

        plt.show()

    
    def test_model(self, initialize_models, iters=5):

        """
            Compare model results against each other

            Provide the initialize_models argument as a function that initializes the models
            that you want to compare. The function should:
            -Only take data as an argument
            -Create a model object from statsmodels.tsa using the data argument
            -Create the implementation of the same model from our libraries
            -The hyperparameters (trend, seasonal, alpha etc..) should be the same
            -Return the created model instances as (<OUR IMPLEMENTATION>, <STATSMODELS MODEL>)

        """

        metrics = {f"sess{i}": {} for i in range(len(list(self.data.keys())))}


        for session, data in enumerate(list(self.data.keys())):

            session_metrics = {}

            session_data = self.data[data]

            imp_outs= []
            sm_outs = []

            for iter in range(iters):

                frac = np.random.uniform(0.2,0.6)

                h = np.random.randint(3,8)

                samp = session_data[:int(len(session_data)*frac)]

                model1, model2 = initialize_models(samp)


                model1.fit()

                fitted = model2.fit()

                implementation_forecasts = model1.predict(h)
                statsmodels_forecast = torch.tensor(np.array(fitted.forecast(h))).reshape(implementation_forecasts.shape)

                rmse = Metric.rmse(implementation_forecasts, statsmodels_forecast)
                mae = Metric.mae(implementation_forecasts, statsmodels_forecast)

                session_metrics[iter] = [rmse, mae]

                df_last_vals = np.squeeze(samp[-15:], axis=-1)

                df_last_vals = torch.tensor(df_last_vals)

                imp_fc = torch.cat((df_last_vals, implementation_forecasts))
                sm_fc = torch.cat((df_last_vals, statsmodels_forecast))

                imp_outs.append(imp_fc)
                sm_outs.append(sm_fc)
                #plot_pred_vs_label(imp_fc, sm_fc)
            
            print(f"Plotting Results for: {data}")
            self.__plot_session(imp_outs, sm_outs)

            metrics[f"sess{session}"]["rmse"] = np.mean(np.array([session_metrics[it][0] for it in session_metrics]))
            metrics[f"sess{session}"]["mae"] = np.mean(np.array([session_metrics[it][1] for it in session_metrics]))

        mean_rmse = np.mean(np.array([metrics[sess]["rmse"] for sess in metrics]))
        mean_mae = np.mean(np.array([metrics[sess]["mae"] for sess in metrics]))

        print("Mean Errors Between Predictions:")
        print(f"Mean RMSE: {mean_rmse}\nMean MAE: {mean_mae}")
                