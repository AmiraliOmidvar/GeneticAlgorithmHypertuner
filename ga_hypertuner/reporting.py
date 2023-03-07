import numpy as np
import pandas as pd
import sys


class Reporting:
    @staticmethod
    def progress(done, pop_size):
        """
        Displays the progress of the current generation.
        :param done: individuals gone through mutation.
        :type done: int

        :param pop_size: total population size
        :type pop_size: int

        :return: None
        """

        prefix = "Generated Pop"
        prog = (prefix + " " + str(done) + "/" + str(pop_size))
        sys.stdout.write('\r' + prog)

    @staticmethod
    def verbose1(scores, score_name, best_params):
        """
        Prints the maximum, minimum, and mean of the given scores along with the best parameters obtained so far during the optimization process.
        :param scores: A NumpyArray of scores obtained for the individuals in the current generation.
        :type scores: NumpyArray

        :param score_name: The scoring criteria that the algorithm tries to optimize. Accepted values are scores that scikit cross validation accepts.
        :type score_name: str

        :param best_params: a dictionary containing current hyperparameters corresponding to best score.
        :return: None
        """
        print("\nMax " + score_name + " : " + str(max(scores)), "Min " + score_name + " : " + str(min(scores)),
              "Mean " + score_name + " : " + str(scores.mean()))
        print("\n" + str(best_params))

    @staticmethod
    def verbose2(vectors):
        """
        This method takes in a list of individuals as input and prints a summary of the hyperparameters values of whole generation.
        :param vectors: A list of dictionaries containing the hyperparameters and corresponding scores of each individual in the population (score of model).
        :type vectors: dict
        :return: None
        """
        vectors_no_score = [d["params"] for d in vectors]
        vectors = pd.DataFrame(vectors_no_score)
        print("-" * 50)
        print("\nParam Values Summary")
        print(vectors.describe())

    @staticmethod
    def verbose3(vectors, score_name):
        """
        This method takes in a list of individuals as input and prints value of hyperparameters and scores of all individuals in population.
        :param vectors: A list of dictionaries containing the hyperparameters and corresponding scores of each individual in the population (score of model).
        :type vectors: dict

        :param score_name: The scoring criteria that the algorithm tries to optimize. Accepted values are scores that scikit cross validation accepts.
        :type score_name: str

        :return: None
        """
        vectors_no_score = [d["params"] for d in vectors]
        vectors_score = [d["score"] for d in vectors]
        score = pd.DataFrame(vectors_score)
        vectors = pd.DataFrame(vectors_no_score)
        vectors[score_name] = score
        print("-" * 50)
        print("\nPopulation")
        print(vectors)
