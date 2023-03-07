from sklearn import datasets
from ga_hypertuner.tuner import Tuner
from sklearn.linear_model import LogisticRegression as lr
from xgboost import XGBRegressor as xgbr

# Example 1
# loading data
x_train, y_train = datasets.load_iris(return_X_y=True, as_frame=True)

# creating model parameters, the solver and class_weight hyperparameters are static and won't change, the C parameter
# is the parameter that algorithm tries to optimize
model_parameters = {"solver": "liblinear", "class_weight": "balanced", "C": [None, float]}

# setting boundaries for parameter C
boundaries = {"C": [0, 1]}

# tuning the C parameter, the scoring criteria is accuracy.
# stratified cross validation is set to true and number of folds of cross validation is set to 3.
best_params = Tuner.tune(x_train, y_train, lr, Tuner.default_ga_parameters, model_parameters
           , boundaries, 'accuracy', stratified=True, k=3, verbosity=1)
print(best_params)

