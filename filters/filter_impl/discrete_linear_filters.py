from abc import ABC

import numpy as np
from sympy.printing.tree import print_node

from filters.filter_impl.filter_base import FilterBase
import control
import dataclasses


@dataclasses.dataclass
class DiscreteLinearKalmanFilter(FilterBase, ABC):
    system: control.LinearIOSystem | control.StateSpace
    convariance_matrix: np.ndarray
    forcing_convariance: np.ndarray
    x_plus: np.ndarray = dataclasses.field(init=False)
    p_k : np.ndarray = dataclasses.field(init=False)

    def __post_init__(self):
        self.p_k = 100000*np.ones_like(self.convariance_matrix)
        self.x_plus = np.zeros(len(self.system.A))

    def __covaraince_update(self, p_k_prev_plus: np.ndarray) -> np.ndarray:
        return self.system.A @ p_k_prev_plus @ self.system.A.T + self.convariance_matrix

    def __covariance_prop(self, p_minus: np.ndarray, k_k: np.ndarray) -> np.ndarray:
        i = np.eye(len(self.system.A))
        exp_1: np.ndarray = (i - k_k @ self.system.C)
        return exp_1 @ p_minus @ exp_1.T + k_k @ self.forcing_convariance @ k_k.T

    def __kalman_gain(self, p_apriori: np.ndarray) -> np.ndarray:
        expr_1: np.ndarray = p_apriori @ self.system.C.T
        expr_2: np.ndarray = (self.system.C @ p_apriori @ self.system.C.T + self.forcing_convariance)
        return expr_1  @ np.linalg.inv(expr_2)
    def __state_update(self, u: np.ndarray) -> np.ndarray:
        return self.system.A @ self.x_plus  + self.system.B @ u

    def __state_prop(self, y_measurement: np.ndarray, k_k: np.ndarray) -> np.ndarray:
        return self.x_plus + k_k @ (y_measurement - self.system.C @ self.x_plus)


    def update(self, u: np.ndarray) -> np.ndarray:
        self.x_plus = self.__state_update(u)
        self.p_k = self.__covaraince_update(self.p_k)


    def step(self, y_measurement: np.ndarray) -> np.ndarray:
        k_k = self.__kalman_gain(p_apriori=self.p_k)
        self.p_k = self.__covariance_prop(self.p_k, k_k)
        self.x_plus = self.__state_prop(y_measurement, k_k)
        return self.x_plus.copy()

    def evaluate(self,y_measurement: np.ndarray, u: np.ndarray) -> np.ndarray:
        self.update(u)
        return self.step(y_measurement)