import dataclasses

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from recursive_filters.naive_recursive_fiiter import Filter, NaiveFilter

matplotlib.use("TkAgg")
plt.rcParams.update({"font.size": 7})
plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')


@dataclasses.dataclass
class NaivePlotterExperiment:
    filter: Filter

    def run_simple_experiment(self) -> None:
        test_sequence = np.ones(shape=(1000, 2)) + np.random.normal(loc=1, scale=2, size=(1000, 2))
        filter_results: np.ndarray = np.array(list(self.filter.consume_signal(test_sequence)))
        fig: Figure = plt.figure(figsize=(16 // 2, 9 // 2))
        states = [f"state_{i}" for i in range(filter_results.shape[1])]
        for i, state in enumerate(states):
            ax: Axes = fig.add_subplot(filter_results.shape[1], 1, i + 1)
            ax.scatter(np.arange(len(test_sequence[:, i])), test_sequence[:, i],
                       s=0.7, color=f"C{i}", label=f"{state}_unfiltered")
            ax.plot(filter_results[:, i], color=f"C{i + 2}", linewidth=0.8, label=f"{state}_filtered")
        fig.set_dpi(500)
        plt.tight_layout()
        plt.show()


def perform_naive_plotting_experiment():
    experiment = NaivePlotterExperiment(
        filter=NaiveFilter(
            x_states=2,
            y_states=2,
            initial_val=np.array([10, 10])
        )
    )

    experiment.run_simple_experiment()


if __name__ == "__main__":
    perform_naive_plotting_experiment()
