import random
import numpy as np
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold
from ga_hypertuner.reporting import Reporting
from ga_hypertuner.visualization import Visualize
import sys
from typing import Union


class GA:
    """
    Genetic algorithm class. All calculations related to the algorithm are methods of this class.

    :param ga_parameters: Parameters of the genetic algorithm. For more information on the parameters, see below.
    :type ga_parameters: dict

    :ga_parameters:
        * *direction* (``int``): Determines whether the score of models should be maximized or minimized. Accepted values are ["max","min"].
        * *pop_size* (``int``): Size of population in each generation. Increasing this value will reduce the chance of local optima. Accepted values are integers greater than 5. Default is 20.
        * *gmax* (``int``): Maximum number of generations. After this many generations, the algorithm will stop and return the best params. Accepted values are integers greater than 1. Default is 50.
        * *fscale* (``int``): A scaling factor that controls the amount of effect that differences between parameters of population members have. larger values will result in larger convergence rate.
        When convergence rate is higher, it will take less time for algorithm to reach local optimum, but the local optimum have lesser chance of being global. Reducing it will opposite result Accepted values are floats between 0 and 1. Default is 0.5.
        * *cp* (``int``): The probability that a child will inherit a parameter from a parent instead of a trial vector. Accepted values are floats between 0 and 1. Default is 0.5.

    :param model_class: Model class that its hyperparameters are being optimized. Any model class that scikit cross-validate module can accept.

    :param model_parameters: hyperparameters that are being optimized. This is a dictionary with parameters of the machine learning model as keys and a list either like [None, Parameter Type] (for optimization of parameter) or [Static Value, Parameter] (for passing the parameter as a static value that will not be changed).
    :type model_parameters: dict

    :param boundaries: Boundary search for hyperparameters. This is a dictionary with parameters of the machine learning model as keys and a list like [start,end].
    :type boundaries: dict

    :param x_train: Training features for the given model. This data will be used to train the model without slicing or sampling.
    :type x_train: Dataframe

    :param y_train: Training target for the given model. This data will be used to train the model without slicing or sampling.
    :type y_train: Dataframe

    :param scoring: The scoring criteria that the algorithm tries to optimize. Accepted values are scores that scikit cross validation accepts.
    :type scoring: str

    :param stop_criteria: Whether the algorithm should stop if it reaches a certain value or not.
    :type stop_criteria: bool

    :param stop_value: The score that, when reached, the algorithm will stop. Default is None.
    :type stop_value:Union[int, float]

    :param k: Number of splits for k-fold cross validation. Accepted values are integers greater than 1. Default is 5.
    :type k: int

    :param stratified: Whether to use stratified cross validation or not. Default is False.
    :type stratified: bool

    :param verbosity: Determines the amount of information that is returned after each generation is generated. Accepted values are 0, 1, 2, or 3. Default is 1.
    :type verbosity: int

    :param show_progress_plot: Whether the progress plot of the score for each generation should be shown at the end of each generation.
    :type show_progress_plot: bool

    :param plot_step: number of generations to skip before displaying progress plot.
    :type plot_step: int
    """

    def __init__(self, ga_parameters: dict, model_class
                 , model_parameters: dict
                 , boundaries: dict
                 , x_train, y_train
                 , scoring
                 , stop_criteria: bool = False, stop_value: Union[int, float] = None
                 , k: int = 5, stratified: bool = False
                 , verbosity: int = 1
                 , show_progress_plot: bool = False
                 , plot_step: int = 1):

        self.generation = 0
        self.gp = ga_parameters
        self.model_class = model_class
        self.mpi = list(model_parameters.values())
        self.mp = list(model_parameters.keys())
        self.dim = len(model_parameters)
        self.s = scoring
        self.b = boundaries
        self.x_t = x_train
        self.y_t = y_train
        self.stop_criteria = stop_criteria
        self.stop_value = stop_value
        self.k = k
        self.stratified = stratified
        self.verbosity = verbosity
        self.show_progress_plot = show_progress_plot
        self.plot_step = plot_step
        self.max_scores = []
        self.min_scores = []
        self.mean_scores = []
        self.best_params = []

    def score(self, params):

        """
        calculates score of an individual. the score of an individual is its models mean score of cross validation.
        :param params: attributes of individual. (hyperparameters)
        :type params: dict
        :return: score of an individual
        """
        model = self.model_class(**params)
        if self.stratified:
            cv = StratifiedKFold(n_splits=self.k, shuffle=True)
        else:
            cv = KFold(n_splits=self.k, shuffle=True)
        score = cross_validate(model, self.x_t, self.y_t, cv=cv, scoring=self.s, return_train_score=False)["test_score"]
        return score.mean()

    def initiation(self):

        """
        Initializes the population vectors with random hyperparameters, and their respective scores.
        :return: A list of population vectors, where each vector is a dictionary containing the hyperparameters as "params" and their respective scores as "score".
        :rtype: list
        """

        self.generation = 1
        vectors = []
        for i in range(self.gp["pop_size"]):
            params = {}
            for j in range(self.dim):
                p = self.mp[j]
                pi = self.mpi[j]
                if type(pi) == list:
                    bound = self.b[p]
                    if pi[1] == int:
                        x = int(random.randint(bound[0], bound[1]))
                    if pi[1] == float:
                        x = random.uniform(bound[0], bound[1])
                else:
                    x = pi
                params[p] = x
            score = self.score(params)
            vector = {"params": params, "score": score}
            vectors.append(vector)
        return vectors

    def mutation(self, vectors):
        """
        Performs mutation on the input list of vectors and returns the updated list of vectors.
        :param vectors: A list of dictionaries containing the hyperparameters and corresponding scores of each individual in the population (score of model).
        :returns: A list of updated dictionaries containing the hyperparameters and corresponding scores of each individual in the population (score of model) after mutation.

        """
        # For each individual in the population as a parent:
        for i in range(self.gp["pop_size"]):
            parent = vectors[i]

            # select three different individuals (different from parent)
            range_values = [x for x in range(0, self.gp["pop_size"] - 1) if x != i]
            chosen = list(random.sample(range_values, k=3))

            # generate a trial individual by adding the difference between the hyperparameters of
            # the two randomly selected individuals (chosen[1] and chosen[2]) multiplied by a scaling factor
            # (gp["fscale"]) to the hyperparameters of the first randomly selected individual (chosen[0]).
            # If the trial parameter is out of bounds, clip it to the nearest bound.
            trial_params = {}
            for j in range(self.dim):
                p = self.mp[j]
                pi = self.mpi[j]
                if type(pi) == list:

                    if pi[1] == int:
                        x = int(vectors[chosen[0]]["params"][p] + self.gp["fscale"] * (
                                vectors[chosen[1]]["params"][p] - vectors[chosen[2]]["params"][p]))

                        if x > self.b[p][1]:
                            x = self.b[p][1]
                        if x < self.b[p][0]:
                            x = self.b[p][0]

                    elif pi[1] == float:
                        x = float(vectors[chosen[0]]["params"][p] + self.gp["fscale"] * (
                                vectors[chosen[1]]["params"][p] - vectors[chosen[2]]["params"][p]))

                        if x > self.b[p][1]:
                            x = self.b[p][1] - 1e-10
                        if x < self.b[p][0]:
                            x = self.b[p][0] + 1e-10
                else:
                    x = pi

                trial_params[p] = x

            # Create a child from trial and parent individuals.
            vectors[i] = self.recombination(parent, trial_params)
            if self.verbosity >= 1:
                Reporting.progress(i + 1, self.gp["pop_size"])
        return vectors

    def recombination(self, parent, trial_params):
        """
        Recombine parent and trial individual to create child, then decides child or parent should be returned to population based on their score.
        :param parent: parent individual
        :param trial_params: trial individual
        :return A list of updated dictionaries containing the hyperparameters and corresponding scores of each individual in the population (score of model) after recombination.:
        """

        child_params = {}
        for i in range(self.dim):
            p = self.mp[i]
            rp = random.uniform(0, 1)
            if rp < self.gp["cp"]:
                child_params[p] = trial_params[p]
            else:
                child_params[p] = parent["params"][p]
        child_score = self.score(child_params)
        child = {"params": child_params, "score": child_score}

        if self.gp["direction"] == "min":
            if child["score"] <= parent["score"]:
                return child
            else:
                return parent
        if self.gp["direction"] == "max":
            if child["score"] >= parent["score"]:
                return child
            else:
                return parent

    def stop(self, scores):
        """
        Checks if max score is reached the specified stop_value
        :param scores: A NumpyArray of scores obtained for the individuals in the current generation.
        :type scores: NumpyArray
        :return: A bool determining whether of algorithm should stop or not
        """
        if self.gp["direction"] == "max":
            if max(scores) > self.stop_value:
                return True
        if self.gp["direction"] == "min":
            if max(scores) < self.stop_value:
                return True

        return False

    def reporting(self, scores, vectors):
        """
        Reports information about the optimization progress based on the specified verbosity level and options.

        :param scores: A NumpyArray of scores obtained for the individuals in the current generation.
        :type scores: NumpyArray

        :param vectors: A list of dictionaries containing the hyperparameters and corresponding scores of each individual in the population.
        :type vectors: list

        :return: None

        """

        # Print verbose information based on verbosity level
        if self.verbosity >= 1:
            Reporting.verbose1(scores, self.s, self.best_params)
        if self.verbosity >= 2:
            Reporting.verbose2(vectors)
        if self.verbosity >= 3:
            Reporting.verbose3(vectors, self.s)

        # Show progress plot if enabled
        if self.generation % self.plot_step == 0 and self.show_progress_plot:
            Visualize.progress_band(self.max_scores, self.min_scores, self.mean_scores, self.s)

    def main(self):
        """
        The main function is the core of the differential evolution algorithm. It initializes the population and performs mutations in each generation, and returns the best parameter set found during the search.

        :return: a dict containing best hyperparameters.

        """
        # initiate the first population
        vectors = self.initiation()
        # while max generation number is not reached
        while self.generation < self.gp["gmax"]:

            print("\nGeneration " + str(self.generation))
            scores = np.zeros(self.gp["pop_size"])
            self.generation += 1

            # mutate the individual
            vectors = self.mutation(vectors)
            for j in range(len(vectors)):
                scores[j] = vectors[j]['score']

            # determine best params
            if self.gp["direction"] == "max":
                self.best_params = vectors[list(scores).index(max(scores))]["params"]
            if self.gp["direction"] == "min":
                self.best_params = vectors[list(scores).index(min(scores))]["params"]

            # calculate max,mean and min of scores
            self.max_scores.append(max(scores))
            self.min_scores.append(min(scores))
            self.mean_scores.append(scores.mean())
            self.reporting(scores, vectors)
            if self.stop_criteria:
                if self.stop(scores):
                    break
        return self.best_params
