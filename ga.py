import random
import numpy as np
from sklearn.model_selection import KFold, cross_validate
import sys


class GA:
    def __init__(self, ga_parameters: dict, model_func
                 , model_parameters: dict
                 , boundaries: list
                 , x_train, y_train
                 , scoring
                 , stop_criteria: bool = False, stop_value: int = None
                 , k: int = 5, stratified: bool = False
                 , verbosity: int = 1
                 , show_progress_plot: bool = False):

        if stop_criteria:
            if stop_value is None:
                pass
                # TODO raise error
        self.generation = 0
        self.gp = ga_parameters
        self.model_func = model_func
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
        self.return_model = return_model
        self.min_losses = []
        self.mean_losses = []
        self.best_params = []

    def loss(self, params):
        model = self.model_func(**params)
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
            loss = self.loss(params)
            vector = {"params": params, "loss": loss}
            vectors.append(vector)
        return vectors

    def mutation(self, vectors):
        for i in range(self.gp["pop_size"]):
            parent = vectors[i]
            chosen = []
            while len(chosen) != 3:
                buffer = random.randint(0, self.gp["pop_size"] - 1)
                if buffer not in chosen and i != buffer:
                    chosen.append(vectors[buffer]["params"])
            trial_params = {}
            for j in range(self.dim):
                p = self.mp[j]

                if self.mpi[j][1] == int:
                    x = int(chosen[0][p] + self.gp["fscale"] * (chosen[1][p] - chosen[2][p]))
                if self.mpi[j][1] == float:
                    x = chosen[0][p] + self.gp["fscale"] * (chosen[1][p] - chosen[2][p])

                if x > self.b[p][1]:
                    x = self.b[p][1]
                if x < self.b[p][0]:
                    x = self.b[p][0]
                trial_params[p] = x
            vectors[i] = self.recombination(parent, trial_params)
            if self.verbosity >= 1:
                self.progress(i + 1, self.gp["pop_size"])
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
        child_loss = self.loss(child_params)
        child = {"params": child_params, "loss": child_loss}

        if self.gp["direction"] == "min":
            if child["loss"] <= parent["loss"]:
                return child
            else:
                return parent
        if self.gp["direction"] == "max":
            if child["loss"] >= parent["loss"]:
                return child
            else:
                return parent

    def stop(self, losses):
        if self.gp["direction"] == "max":
            if max(losses) > self.stop_value:
                return True
        if self.gp["direction"] == "min":
            if max(losses) < self.stop_value:
                return True

        return False

    def reporting(self, losses):
        if self.verbosity >= 1:
            self.verbose1(losses, self.s, self.best_params)
        if self.verbosity >= 2:
            self.verbose2(vectors)
        if self.verbosity >= 3:
            self.verbose3(vectors, self.s)
        if self.generation > 1 and self.show_progress_plot:
            progress_band(self.max_losses, self.min_losses, self.mean_losses, self.s)

    def main(self):
        vectors = self.initiation()
        while self.generation < self.gp["gmax"]:
            print("\nGeneration " + str(self.generation))
            losses = np.zeros(self.gp["pop_size"])
            self.generation += 1
            vectors = self.mutation(vectors)
            for j in range(len(vectors)):
                losses[j] = vectors[j]['loss']

            if self.gp["direction"] == "max":
                self.best_params = vectors[list(losses).index(max(losses))]["params"]
            if self.gp["direction"] == "min":
                self.best_params = vectors[list(losses).index(min(losses))]["params"]

            self.max_losses.append(max(losses))
            self.min_losses.append(min(losses))
            self.mean_losses.append(losses.mean())
            self.reporting(losses)
            if self.stop_criteria:
                if self.stop(losses):
                    break
        return vectors
