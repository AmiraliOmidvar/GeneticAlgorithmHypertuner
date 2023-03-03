import matplotlib.pyplot as plt


class Visualize:
    @staticmethod
    def progress_band(mins, means, maxs, score_name):
        plt.figure(figsize=(15, 10))
        x = range(1, len(maxs) + 1)
        plt.xticks(x)
        plt.grid()
        plt.title(score_name + " Progress By Generation", fontsize=20)
        plt.ylabel(score_name, fontsize=20)
        plt.xlabel("Generation", fontsize=20)
        plt.plot(x, maxs, label="Max " + score_name, linewidth=4, color="blue")
        plt.plot(x, means, label="Mean " + score_name, linewidth=4, color="#9C27B0")
        plt.plot(x, mins, label="Min " + score_name, linewidth=4, color="red")
        plt.fill_between(x, maxs, mins, color="#9C27B0", alpha=0.2)
        plt.legend(loc="upper left", prop={'size': 12})
        plt.show()


Visualize.progress_band([8, 9, 11, 12], [9, 10, 11.5, 12.3], [11, 12, 12, 12.5], "r2")
