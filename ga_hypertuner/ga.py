import random
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from ga_hypertuner.reporting import Reporting
from ga_hypertuner.visualization import Visualize
import sys


class GA:
    def __init__(self, ga_parameters: dict, model_class
                 , model_parameters: dict
                 , boundaries: dict
                 , x_train, y_train
                 , scoring
                 , stop_criteria: bool = False, stop_value: int = None
                 , k: int = 5, stratified: bool = False
                 , verbosity: int = 1
                 , show_progress_plot: bool = False):

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
        self.max_scores = []
        self.min_scores = []
        self.mean_scores = []
        self.best_params = []

    def score(self, params):
        model = self.model_class(**params)
        kf = KFold(n_splits=self.k, shuffle=True)
        score = cross_validate(model, self.x_t, self.y_t, cv=kf, scoring=self.s, return_train_score=False)["test_score"]
        return score.mean()

    def initiation(self):
        self.generation = 1
        vectors = []
        for i in range(self.gp["pop_size"]):
            params = {}
            for j in range(self.dim):
                p = self.mp[j]
                pi = self.mpi[j]
                if pi[0] is None:
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
        for i in range(self.gp["pop_size"]):
            parent = vectors[i]
            range_values = [x for x in range(0, self.gp["pop_size"] - 1) if x != i]
            chosen = list(random.sample(range_values, k=3))
            trial_params = {}
            for j in range(self.dim):
                p = self.mp[j]

                if self.mpi[j][1] == int:
                    x = int(vectors[chosen[0]]["params"][p] + self.gp["fscale"] * (vectors[chosen[1]]["params"][p] - vectors[chosen[2]]["params"][p]))
                if self.mpi[j][1] == float:
                    x = int(vectors[chosen[0]]["params"][p] + self.gp["fscale"] * (vectors[chosen[1]]["params"][p] - vectors[chosen[2]]["params"][p]))

                if x > self.b[p][1]:
                    x = self.b[p][1]
                if x < self.b[p][0]:
                    x = self.b[p][0]
                trial_params[p] = x
            vectors[i] = self.recombination(parent, trial_params)
            if self.verbosity >= 1:
                Reporting.progress(i + 1, self.gp["pop_size"])
        return vectors

    def recombination(self, parent, trial_params):
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
        if self.gp["direction"] == "max":
            if max(scores) > self.stop_value:
                return True
        if self.gp["direction"] == "min":
            if max(scores) < self.stop_value:
                return True

        return False

    def reporting(self, scores, vectors):
        if self.verbosity >= 1:
            Reporting.verbose1(scores, self.s, self.best_params)
        if self.verbosity >= 2:
            Reporting.verbose2(vectors)
        if self.verbosity >= 3:
            Reporting.verbose3(vectors, self.s)
        if self.generation > 1 and self.show_progress_plot:
            Visualize.progress_band(self.max_scores, self.min_scores, self.mean_scores, self.s)

    def main(self):
        vectors = self.initiation()
        while self.generation < self.gp["gmax"]:
            print("\nGeneration " + str(self.generation))
            scores = np.zeros(self.gp["pop_size"])
            self.generation += 1
            vectors = self.mutation(vectors)
            for j in range(len(vectors)):
                scores[j] = vectors[j]['score']

            if self.gp["direction"] == "max":
                self.best_params = vectors[list(scores).index(max(scores))]["params"]
            if self.gp["direction"] == "min":
                self.best_params = vectors[list(scores).index(min(scores))]["params"]

            self.max_scores.append(max(scores))
            self.min_scores.append(min(scores))
            self.mean_scores.append(scores.mean())
            self.reporting(scores, vectors)
            if self.stop_criteria:
                if self.stop(scores):
                    break
        return self.best_params
