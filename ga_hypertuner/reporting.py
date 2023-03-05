import numpy as np
import pandas as pd


class Reporting:
    @staticmethod
    def progress(done, pop_size):
        prefix = "Generated Pop"
        prog = (prefix + " " + str(done) + "/" + str(pop_size))
        sys.stdout.write('\r' + prog)

    @staticmethod
    def verbose1(losses, score_name, best_params):
        print("\nMax " + score_name + " : " + str(max(losses)), "Min " + score_name + " : " + str(min(losses)),
              "Mean " + score_name + " : " + str(losses.mean()))
        print("\n"+best_params)

    @staticmethod
    def verbose2(vectors):
        vectors_no_loss = [d["params"] for d in vectors]
        vectors = pd.DataFrame(vectors_no_loss)
        print("-"*50)
        print("\nParam Values Summary")
        print(vectors.describe())

    @staticmethod
    def verbose3(vectors, score_name):
        vectors_no_loss = [d["params"] for d in vectors]
        vectors_loss = [d["loss"] for d in vectors]
        loss = pd.DataFrame(vectors_loss)
        vectors = pd.DataFrame(vectors_no_loss)
        vectors[score_name] = loss
        print("-" * 50)
        print("\nPopulation Size")
        print(vectors)

