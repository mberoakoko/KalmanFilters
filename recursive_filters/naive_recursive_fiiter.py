import dataclasses
import numpy as np
import abc
from typing import Generator


class Filter(abc.ABC):
    @abc.abstractmethod
    def consume_sample(self, sample: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def consume_signal(self, signal: np.ndarray) -> Generator[np.ndarray, None, None]:
        yield from (self.consume_sample(y_k) for y_k in signal)


@dataclasses.dataclass
class NaiveFilter(Filter):
    x_states: int
    y_states: int
    initial_val: np.ndarray
    p: np.ndarray = dataclasses.field(default=np.array([]), init=False)
    __x: np.ndarray = dataclasses.field(default=np.array([]), init=False)

    def __post_init__(self):
        self.h = np.eye(self.x_states)
        self.p = 10000000 * np.eye(self.x_states)
        self.r = 0.1 * np.eye(self.y_states)
        self.__x: np.ndarray = self.initial_val.copy()

    def consume_sample(self, sample: float) -> np.ndarray:
        k: np.ndarray = self.p @ self.h.T @ np.linalg.inv(self.h @ self.p @ self.h.T + self.r)
        self.__x = self.__x + k @ (sample - self.h @ self.__x)
        i = np.identity(self.x_states)
        self.p = (i - k @ self.h) @ self.p @ (i - k @ self.h).T + k @ self.r @ k.T
        return self.__x.copy()

    def consume_signal(self, signal: np.ndarray) -> Generator[np.ndarray, None, None]:
        yield from (self.consume_sample(y_k) for y_k in signal)


if __name__ == "__main__":
    naive_recursive_filter = NaiveFilter(
        x_states=2,
        y_states=2,
        initial_val=np.array([0, 0])
    )
    test_sequence = np.ones(shape=(1000, 2)) + np.random.normal(loc=1, scale=2, size=(1000, 2))
    for x_k_hat in naive_recursive_filter.consume_signal(test_sequence):
        print(x_k_hat)
