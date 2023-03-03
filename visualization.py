import matplotlib.pyplot as plt


class Visualize:
    @staticmethod
    def progress_band(maxs, mins, means, score_name):
        print("-" * 50)
        plt.figure(figsize=(8, 4))
        x = range(1, len(maxs) + 1)
        plt.xticks(x)
        plt.grid()
        plt.title(score_name + " Progress By Generation", fontsize=15)
        plt.ylabel(score_name, fontsize=15)
        plt.xlabel("Generation", fontsize=15)
        plt.plot(x, maxs, label="Max " + score_name, linewidth=2, color="blue")
        plt.plot(x, means, label="Mean " + score_name, linewidth=2, color="#9C27B0")
        plt.plot(x, mins, label="Min " + score_name, linewidth=2, color="red")
        plt.fill_between(x, maxs, mins, color="#9C27B0", alpha=0.2)
        plt.legend(loc="upper left", prop={'size': 8})
        plt.show()
