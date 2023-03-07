from sklearn import datasets
from ga_hypertuner.tuner import Tuner
from xgboost import XGBRegressor as xgbr

# Example 2
# loading data
x_train, y_train = datasets.load_diabetes(return_X_y=True, as_frame=True)


# setting genetic algorithm parameters
ga_parameters = {"pop_size": 15, "fscale": 0.6, "gmax": 200, "direction": "max", "cp": 0.6}

# setting xgboost model parameters
model_parameters = {"eta": [None, float], "min_child_weight": [None, float], "colsample_bytree": [None, float],
                    "n_estimators": 100, "alpha": [None, float], "gamma": [None, float]}

# setting boundaries for parameters
boundaries = {"eta": [0, 1], "min_child_weight": [0, 5], "colsample_bytree": [0, 1],
              "alpha": [0, 1], "gamma": [0, 1]}

# tuning parameters, a progress plot will be displayed every 10 generation
best_params = Tuner.tune(x_train, y_train, xgbr, ga_parameters, model_parameters
           , boundaries, 'r2', k=3, verbosity=3, show_progress_plot=True, plot_step=10)
