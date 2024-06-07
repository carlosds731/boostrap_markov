import typing
from itertools import cycle

colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
import numpy as np
import matplotlib.pyplot as plt
import os


class MarkovChain:
    FIXED_NUMBER_BLOCKS = "fixed_number_blocks"

    REGENERATION_BASED_BOOTSTRAP = "regeneration_bootstrap"

    # region Internal data
    _path: typing.Optional[np.ndarray] = None

    _most_visited: typing.Optional[int] = None

    _step_n: typing.Optional[int] = None

    _random_seed: typing.Optional[int] = None

    # region histogram internal data
    _n: typing.Optional[np.ndarray] = None

    _bins: typing.Optional[np.ndarray] = None

    _patches = None

    # endregion histogram internal data

    # region Records internal data
    _index_records: typing.Optional[np.ndarray] = None

    _record_values: typing.Optional[np.ndarray] = None

    # Dictionary of regeneration blocks (the key is the state)
    _regeneration_blocks = None

    # endregion Records internal data

    # endregion Internal data

    # region public interface

    def __init__(self, step_n: int, name: str, random_seed=None) -> None:
        self._step_n = step_n
        self.name = name
        self._random_seed = random_seed

    def get_path(self) -> np.ndarray:
        """
        Returns the path of the Markov Chain. If it hasn't been generated yet, it generates it.

        Returns:
            np.ndarray: The path of the Markov Chain.
        """
        if self._path is None:
            self.generate_path()
        return self._path

    def get_step_n(self) -> int:
        """
        Returns the number of steps in the Markov Chain.

        Returns:
            int: The number of steps in the Markov Chain.
        """
        return len(self._path) - 1

    def get_most_visited_state(self) -> int:
        """
        Returns the most visited state in the Markov Chain.

        Returns:
            int: The most visited state.
        """
        if self._most_visited is not None:
            return self._most_visited
        self._most_visited = self._calculate_most_visited()
        return self._most_visited

    def get_number_visits_until_time(
        self, a: int, b: int, j: typing.Optional[int] = None
    ) -> int:
        """
        Returns the number of visits to states between a and b (inclusive) until time j.

        Args:
            a (int): The lower bound of the states to consider.
            b (int): The upper bound of the states to consider.
            j (int, optional): The time until which to consider visits. If None, considers all times.

        Returns:
            int: The number of visits to states between a and b until time j.
        """
        path = self.get_path()
        if j:
            return np.sum((a <= path[: j + 1]) & (path[: j + 1] <= b))
        else:
            return np.sum((a <= path) & (path <= b))

    def get_times_of_visits_until_time(
        self, a: int, b: int, j: typing.Optional[int] = None
    ) -> np.ndarray:
        """
        Returns the times of visits to states between a and b (inclusive) until time j.

        Args:
            a (int): The lower bound of the states to consider.
            b (int): The upper bound of the states to consider.
            j (int, optional): The time until which to consider visits. If None, considers all times.

        Returns:
            np.ndarray: The times of visits to states between a and b until time j.
        """
        path = self.get_path()
        if j:
            return np.where((a <= path[: j + 1]) & (path[: j + 1] <= b))[0]
        else:
            return np.where((a <= path) & (path <= b))[0]

    def get_time_of_first_visit(self, state):
        """
        Returns the time of the first visit to a given state
        """
        return np.argmax(self._path == int(state))

    def get_times_of_visits(self, state=None):
        """
        Returns the times of visits to a given state
        """
        if state is None:
            state = self.get_most_visited_state()
        return [x[0] for x in np.argwhere(self._path == int(state))[0:]]

    def get_regeneration_block_sizes(
        self, state: typing.Optional[int] = None
    ) -> np.ndarray:
        """
        Returns the sizes of the regeneration blocks for a given state.

        Args:
            state (int, optional): The state to consider. If None, uses the most visited state.

        Returns:
            np.ndarray: The sizes of the regeneration blocks for the given state.
        """
        if state is None:
            state = self.get_most_visited_state()
        _visit_times = self.get_times_of_visits(state)
        return np.ediff1d(_visit_times)

    def get_times_of_visit_and_regeneration_block_sizes(
        self, state: typing.Optional[int] = None
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Returns the times of visits and sizes of the regeneration blocks for a given state.

        Args:
            state (int, optional): The state to consider. If None, uses the most visited state.

        Returns:
            tuple of np.ndarray: The times of visits and sizes of the regeneration blocks for the given state.
        """
        if state is None:
            state = self.get_most_visited_state()
        _visit_times = self.get_times_of_visits(state)
        return _visit_times, np.ediff1d(_visit_times)

    def get_number_of_regeneration_blocks(
        self, state: typing.Optional[int] = None
    ) -> int:
        """
        Returns the number of regeneration blocks for a given state.

        Args:
            state (int, optional): The state to consider. If None, uses the most visited state.

        Returns:
            int: The number of regeneration blocks for the given state.
        """
        return len(self.get_regeneration_blocks(state=state))

    def get_regeneration_blocks(
        self, state: typing.Optional[int] = None
    ) -> typing.List[np.ndarray]:
        """
        Returns the regeneration blocks for a given state.

        Args:
            state (int, optional): The state to consider. If None, uses the most visited state.

        Returns:
            list of np.ndarray: The regeneration blocks for the given state.
        """
        _visit_times = self.get_times_of_visits(state=state)
        if self._regeneration_blocks:
            try:
                return self._regeneration_blocks[state]
            except KeyError:
                self._regeneration_blocks = dict()
        else:
            self._regeneration_blocks = dict()
        blocks = list()
        _path = self.get_path()

        for index in range(0, len(_visit_times) - 1):
            blocks.append(_path[_visit_times[index] + 1 : _visit_times[index + 1] + 1])

        self._regeneration_blocks[state] = blocks
        return blocks

    def plot_histogram(self) -> None:
        """
        Plots the histogram of the Markov chain.
        """
        self.get_histogram()
        _most_hit = self.get_most_visited_state()
        _step_n = self._step_n
        _max_number_of_hits = self.max_number_of_visits
        first_visit = self.get_time_of_first_visit(_most_hit)
        plt.grid(axis="y", alpha=0.75)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("{0} histogram n={1}".format(self.name, _step_n))
        plt.figtext(
            1.0, 0.4, "Max number of hists {0}".format(int(_max_number_of_hits))
        )
        plt.figtext(1.0, 0.3, "Most visited state is {0}".format(_most_hit))
        plt.figtext(1.0, 0.2, "First visit after {0}".format(first_visit))
        plt.savefig(
            os.path.join(
                "histograms", "Histogram of {0}, n = {1}.png".format(self.name, _step_n)
            ),
            dpi=700,
            bbox_inches="tight",
        )
        plt.show()

    def plot_simulation(
        self,
        state: typing.Optional[int] = None,
        base_path: typing.Optional[str] = None,
        show: bool = True,
        save: bool = True,
        custom_title: typing.Optional[str] = None,
    ) -> None:
        """
        Plots a realization of the Markov chain.

        Args:
            state (int, optional): The state to highlight visits to. If None, uses the most visited state.
            base_path (str, optional): The base path to save the plot to. If None, uses "plots".
            show (bool, optional): Whether to show the plot. Default is True.
            save (bool, optional): Whether to save the plot. Default is True.
            custom_title (str, optional): A custom title for the plot. If None, uses a default title.
        """
        self.get_histogram()
        _step_n = self._step_n
        _path = self._path
        _start = _path[:1]
        _stop = _path[-1:]
        if state is None:
            state = self.get_most_visited_state()
        # Plot the path
        fig = plt.figure(figsize=(8, 4), dpi=200)
        ax = fig.add_subplot(111)
        ax.scatter(np.arange(_step_n + 1), _path, c="blue", alpha=0.25, s=0.05)
        ax.plot(
            _path,
            c="blue",
            alpha=0.5,
            lw=0.5,
            ls="-",
        )
        ax.plot(0, _start, c="red", marker="+")
        ax.plot(_step_n, _stop, c="black", marker="o")
        _times_of_visit = self.get_times_of_visits(state)
        for c_p in _times_of_visit:
            ax.plot(c_p, state, c="green", marker="*")
        if custom_title:
            plt.title(custom_title)
        else:
            plt.title(
                "{0} n={1}, T(n)={2}".format(
                    self.name, _step_n, len(_times_of_visit) - 1
                )
            )
        plt.tight_layout(pad=0)
        if save:
            if not base_path:
                base_path = os.path.join("plots")
            plt.savefig(
                os.path.join(
                    base_path,
                    "Realization of {0} n = {1}.png".format(self.name, _step_n),
                ),
                dpi=250,
            )
        if show:
            plt.show()

    @property
    def max_number_of_visits(self):
        return np.max(self._n)

    def get_histogram(self):
        if self._path is None:
            self.generate_path()
        if self._n is None:
            return self._calculate_histogram()
        return self._n, self._bins, self._patches

    def generate_path(self):
        """
        Generates the _path and stores it
        """
        self._n = None
        self._bins = None
        self._patches = None
        self._most_visited = None
        self._path = self._generate_path()

    def get_regeneration_markov_chain(self, state=None):
        if state is None:
            state = self.get_most_visited_state()
        _visit_times = self.get_times_of_visits(state)
        _reg_path = self.get_path()[_visit_times[0] : _visit_times[-1] + 1]
        _reg_mc = type(self)(
            len(_reg_path) - 1, "Regeneration of {0}".format(self.name)
        )
        _reg_mc._path = _reg_path
        return _reg_mc

    def get_bootstrap_markov_chain(self, state=None, method=FIXED_NUMBER_BLOCKS):
        regeneration_blocks = self.get_regeneration_blocks(state=state)
        length = 0
        _reg_path = None
        t_n = 0
        if method == MarkovChain.FIXED_NUMBER_BLOCKS:
            resampled_blocks = np.random.choice(
                regeneration_blocks, size=len(regeneration_blocks), replace=True
            )
            _reg_path = np.concatenate(tuple(resampled_blocks))
            length = len(_reg_path) - 1
            t_n = len(regeneration_blocks)
        elif method == MarkovChain.REGENERATION_BASED_BOOTSTRAP:
            num_points = 0
            resampled_blocks = list()
            t_n = 0
            while True:
                _block = np.random.choice(regeneration_blocks, 1, replace=True)[0]
                num_points += len(_block)
                if num_points >= self.get_step_n():
                    break
                resampled_blocks.append(_block)
                t_n += 1
            _reg_path = np.concatenate(tuple(resampled_blocks))
            length = len(_reg_path) - 1
        else:
            raise NotImplementedError(
                "method {0} is not implemented yet".format(method)
            )
        _reg_mc = type(self)(length, "Booostrap {0} of {1}".format(method, self.name))
        _reg_mc._path = _reg_path
        return _reg_mc, t_n

    def get_continuous_counting_process(self, state=None):
        if state is None:
            state = self.get_most_visited_state()

        def _wrap(t):
            _slice = self._path[: int(np.floor(self._step_n * t))]
            vistis = np.where(_slice == state)
            count = np.count_nonzero(vistis)
            return count

        return _wrap

    # endregion public interface

    # region internal functions

    def _calculate_histogram(self):
        """
            Calculates the histogram. If the _path has not been generated it does it
        :return: It returns the matplotlib histogram
        :rtype: tuple
        """
        if self._path is None:
            self.generate_path()
        _path = self._path
        n, bins, patches = plt.hist(
            x=_path,
            bins=np.arange(min(_path) - 0.5, max(_path) + 0.5 + 1, 1),
            color="#0504aa",
            alpha=0.7,
            rwidth=0.85,
        )
        self._n = n
        self._bins = bins
        self._patches = patches
        plt.close()
        return n, bins, patches

    def _generate_path(self) -> np.ndarray:
        """
        Generates the _path
        """
        raise NotImplemented("You must implement the _path")

    def _calculate_most_visited(self):
        self.get_histogram()
        max_number_of_hits = self.max_number_of_visits
        # indexes on the histogram
        _indexes = np.where(self._n == max_number_of_hits)[0]
        # The value of the most visited state
        return [int(self._bins[i] + 0.5) for i in _indexes][0]

    # endregion internal functions
