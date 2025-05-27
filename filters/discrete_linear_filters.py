from abc import ABC

import numpy as np

from filter_base import FilterBase
import control
import dataclasses


@dataclasses.dataclass
class DiscreteLinearKalmanFilter(FilterBase, ABC):
    system: control.LinearIOSystem
    convariance_matrix: np.ndarray
    forcing_convariance: np.ndarray
    x_plus: np.ndarray = dataclasses.field(init=False)

    def __covariance_prop(self, p_minus: np.ndarray, k_k: np.ndarray) -> np.ndarray:
        i = np.eye(len(self.system.A))
        exp_1: np.ndarray = (i - k_k @ self.system.C)
        return exp_1 @ p_minus @ exp_1.T + k_k @ self.forcing_convariance @ k_k.T

    def __kalman_gain(self, p_apriori: np.ndarray) -> np.ndarray:
        return p_apriori @ self.system.C @ (self.system.C @ p_apriori @ self.system.C.T + self.forcing_convariance)

    def update(self, y_measurement: np.ndarray) -> np.ndarray:
        ...

    def step(self) -> np.ndarray:
        ...