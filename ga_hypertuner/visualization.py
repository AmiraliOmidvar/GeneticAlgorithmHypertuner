import matplotlib.pyplot as plt


class Visualize:
    @staticmethod
    def progress_band(maxs, mins, means, score_name):
        """
        The Visualize class contains a static method progress_band that generates a line chart of the progress of a score (maximum, minimum, and mean) by generation.
        :param maxs: a list of the maximum scores for each generation.
        :type maxs: list

        :param mins: a list of the minimum scores for each generation.
        :type mins: list

        :param means: a list of the mean scores for each generation.
        :type means: list

        :param score_name: The scoring criteria that the algorithm tries to optimize. Accepted values are scores that scikit cross validation accepts.
        :type score_name: str

        :return: None, but displays a line chart of the score progress by generation.
        """

        print("-" * 50)
        plt.figure(figsize=(8, 4))
        x = range(1, len(maxs) + 1)
        plt.xticks(x)
        plt.grid()
        plt.title(score_name + " Progress By Generation", fontsize=12)
        plt.ylabel(score_name, fontsize=12)
        plt.xlabel("Generation", fontsize=12)
        plt.plot(x, maxs, label="Max " + score_name, linewidth=2, color="blue")
        plt.plot(x, means, label="Mean " + score_name, linewidth=2, color="#9C27B0")
        plt.plot(x, mins, label="Min " + score_name, linewidth=2, color="red")
        plt.fill_between(x, maxs, mins, color="#9C27B0", alpha=0.2)
        plt.legend(loc="upper left", prop={'size': 8})
        plt.show()
