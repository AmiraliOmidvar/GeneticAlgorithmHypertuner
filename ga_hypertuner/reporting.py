import numpy as np
import pandas as pd
import sys


class Reporting:
    @staticmethod
    def progress(done, pop_size):
        prefix = "Generated Pop"
        prog = (prefix + " " + str(done) + "/" + str(pop_size))
        sys.stdout.write('\r' + prog)

    @staticmethod
    def verbose1(scores, score_name, best_params):
        print("\nMax " + score_name + " : " + str(max(scores)), "Min " + score_name + " : " + str(min(scores)),
              "Mean " + score_name + " : " + str(scores.mean()))
        print("\n" + str(best_params))

    @staticmethod
    def verbose2(vectors):
        vectors_no_score = [d["params"] for d in vectors]
        vectors = pd.DataFrame(vectors_no_score)
        print("-" * 50)
        print("\nParam Values Summary")
        print(vectors.describe())

    @staticmethod
    def verbose3(vectors, score_name):
        vectors_no_score = [d["params"] for d in vectors]
        vectors_score = [d["score"] for d in vectors]
        score = pd.DataFrame(vectors_score)
        vectors = pd.DataFrame(vectors_no_score)
        vectors[score_name] = score
        print("-" * 50)
        print("\nPopulation Size")
        print(vectors)
