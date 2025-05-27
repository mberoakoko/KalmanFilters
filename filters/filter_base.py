import abc
from abc import ABC

import numpy as np


class FilterBase(abc.ABC):

    @abc.abstractmethod
    def step(self) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, y_measurement: np.ndarray) -> np.ndarray:
        raise NotImplementedError
