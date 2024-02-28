# Genetic Algorithm Hypertuner

Genetic Algorithm HyperTuner is a Python package designed to help users fine-tune hyperparameters for their machine-learning models using a genetic algorithm approach. (differential evolution)

## Features
Users can easily specify the hyperparameters they want to optimize, define the range and constraints for each hyperparameter, and set up the genetic algorithm parameters such as population size, combination probability, and mutation rates.

The package offers an intuitive interface that allows users to easily monitor the progress of the optimization process and visualize the results. With GA_HyperTuner, users can save a lot of time and effort that would otherwise be spent on manual hyperparameter tuning. 

By leveraging the power of genetic algorithm (Differential Evolution), GA_HyperTuner can help users find optimal hyperparameters for their models that lead to better accuracy and performance.

## Installation

Use pip to install ga_hypertuner.

```bash
pip install ga_hypertuner
```

## Usage

```python
from ga_hypertuner.tuner import Tuner
from sklearn import datasets
from sklearn.linear_model import LogisticRegression as lr
from xgboost import XGBRegressor as xgbr
##########################################################################
##########################################################################    
# Example 1
# loading data
x_train, y_train = datasets.load_iris(return_X_y=True, as_frame=True)

# creating model parameters, the solver and class_weight hyperparameters are static and won't change the C parameter
# is the parameter that the algorithm tries to optimize
model_parameters = {"solver": "liblinear", "class_weight": "balanced", "C": [None, float]}

# setting boundaries for parameter C
boundaries = {"C": [0, 1]}

# tuning the C parameter, the scoring criteria is accuracy.
# stratified cross-validation is set to true and the number of folds of cross-validation is set to 3.
best_params = Tuner.tune(x_train, y_train, lr, Tuner.default_ga_parameters, model_parameters
           , boundaries, 'accuracy', stratified=True, k=3, verbosity=1)
           
##########################################################################
##########################################################################        
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
##########################################################################
##########################################################################    
```

Output sample:
```
Max r2 : 0.4647872361252694 Min r2 : 0.4106477735163949 Mean r2 : 0.43554044453711327

{'eta': 0.044987761443632285, 'min_child_weight': 3.2289184084919063, 'colsample_bytree': 0.30200366037496074, 'n_estimators': 100, 'alpha': 0.6094719857402888, 'gamma': 0.7681670166330536}
--------------------------------------------------

Param Values Summary
             eta  min_child_weight  colsample_bytree  n_estimators      alpha  \
count  15.000000         15.000000         15.000000          15.0  15.000000   
mean    0.070673          3.262381          0.444616         100.0   0.777078   
std     0.019617          1.275494          0.126947           0.0   0.234755   
min     0.041235          0.927918          0.279023         100.0   0.353341   
25%     0.053972          2.550634          0.329209         100.0   0.667062   
50%     0.072370          3.228918          0.422560         100.0   0.823875   
75%     0.082690          4.389026          0.541351         100.0   1.000000   
max     0.109752          5.000000          0.639734         100.0   1.000000   

           gamma  
count  15.000000  
mean    0.774367  
std     0.174307  
min     0.453010  
25%     0.631866  
50%     0.768167  
75%     0.930546  
max     1.000000  
--------------------------------------------------
```
<p align="center">
  <img src="https://github.com/AmiraliOmidvar/GeneticAlgorithmHypertuner/assets/118000089/d96e013e-b458-4331-b468-5205c5a57136" />
</p>

for more examples and info please refer to doc.

## Documentation

you can find ga_hypertuner [doc here](https://ga-hypertuner.readthedocs.io/en/latest/).

## Contributing

Any contribution is welcome. please open an issue to discuss changes or improvements.

## License

[MIT](https://github.com/AmiraliOmidvar/ga_hypertuner/blob/master/LICENCE.txt)
