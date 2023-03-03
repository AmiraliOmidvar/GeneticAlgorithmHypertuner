import random
import numpy as np
from sklearn.model_selection import KFold, cross_validate


class GA:
    def __init__(self, ga_parameters: dict, model_func
                 , model_parameters: dict
                 , boundaries: list
                 , scoring
                 , x_train, y_train
                 , k: int = 5, stratified: bool = False
                 , verbosity: int = 1
                 , show_progress_plot: bool = False
                 , return_model:bool = False
                 , save_model: bool = True
                 , save_location: str = ""):
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
        self.k = k
        self.stratified = stratified
        self.verbosity = verbosity
        self.show_progress_plot

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
                    bound = self.b[j]
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

                if x > self.b[j][1]:
                    x = self.b[j][1]
                if x < self.b[j][0]:
                    x = self.b[j][0]
                trial_params[p] = x
            vectors[i] = self.recombination(parent, trial_params)
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
            if max(losses) > self.gp["stop_value"]:
                return True
        if self.gp["direction"] == "min":
            if max(losses) < self.gp["stop_value"]:
                return True

        return False

    def tune(self):
        i = 0
        vectors = self.initiation()
        while i < self.gp["gmax"]:
            losses = np.zeros(self.gp["pop_size"])
            i += 1
            vectors = self.mutation(vectors)
            for j in range(len(vectors)):
                losses[j] = vectors[j]['loss']
            print(vectors)
            print("\n")
            print(max(losses), losses.mean())
            print("\n")
            print("\n")
            if self.stop(losses):
                break
        return vectors
